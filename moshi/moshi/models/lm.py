# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from ..modules.streaming import StreamingContainer, StreamingModule
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..utils.compile import CUDAGraphed
from ..utils.sampling import sample_token

logger = logging.getLogger(__name__)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None
        # Unlike for audio, here we authorize the model to output the special token.
        # The text_emb is acting on actual text tokens (vocab 32000)
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        self.depformer.set_streaming_propagate(False)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    # Q: What does it do and why is it always 1?
    # - Does it refer to the sementic vs accoustic codebooks?
    # - No, its seems to refer to text (inner monologue) tokens vs. audio tokens, c.f. `text_emb = self.text_emb(input_sequence[:, 0])`
    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    # 2/0
    def forward_text(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        # 2/1 First compute embedding for each audio codebook sequence[:, 1:]
        # This includes the audio for the input and the output stream (i.e. 16 entries)!
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            # CHECK: this seems to be a more memory efficient way to do this than explicit sum as in audiocraft?
            # Or not, since the gradient computation would still need to keep track of the summands?
            input_ = audio_emb if input_ is None else input_ + audio_emb
        # 2/2 Compute text embedding for the first text token (sequence[:, 0])
        text_emb = self.text_emb(input_sequence[:, 0])
        # Q: Why is the text embedding added to the audio embeddings? Isn't the text *concatenated* with the audio tokens?
        # - It's not fully clear from the paper --> need to reread carefully
        # - Maybe the multi-stream means actually that streams are summed internally (e.g. as done for the different codebook streams)
        input_ = text_emb if input_ is None else input_ + text_emb
        transformer_out = self.transformer(input_)

        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        # 2/3 based on the pre-logits compute text_logits (just a linear head)
        # So transformer_out is a combined representation of text and audio tokens
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    # 4/0
    def forward_depformer(
        self,
        depformer_cb_index: int,  # index of the codebook for which we generate the token (first index text token, then the audio tokens)
        sequence: torch.Tensor,  
        # sequence of tokens for each codebook (generated sequentially, by repated calls to this function)
        # from the depformer_step, this seems only the prev. token (and not the whole sequence) is passed in here
        # this is confirmed by the assert checks below
        # CHECK: how this would be done for training
        # - Do we apply this forward function once on the whole (ground-truth) sequence?
        transformer_out: torch.Tensor,  # output of the temporal transformer (not chagned)
    ) -> torch.Tensor:
        
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        # 4/1 First, we apply only a linear layer (optionally a different one for each codebook) on the input.
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        # 4/2 Then, we compute the embedding for the codebook sequence (prev. generated codebook token).
        # QUESTION: what is the text embedding for the first cb_index?
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        # 4/3 The final input is compute by summing the input sequence (i.e. prev. token) embedding an
        # the temporal transformer output embedding (different for each cb_index)
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        
        # 4/4 Compute the output by applying the forward function and final linear output projection (again spefific to cb_index)
        # This is the same as in musicgen
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits


@dataclass
class _LMGenState:
    cache: torch.Tensor
    initial: torch.Tensor  # this is used to init the cache befare offset >= delay
    # /STREAM: why do we track state of these graphed functions?
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        # /CACHE Why don't we delete the cache?
        # - Since it's overwritten in the _init_streaming_state anyway
        # - But this seems to be not fully consistent with the idea behind the reset function
        self.offset = 0


# This seems to be the api class for the LMModel
class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        # CACHE/ This is determines the size of the chache --> why is it so small?
        # - It's only used to store the delay pattern of "unprocessed" tokens
        # - The real cache is the KV cache in the transformer

        # first fill the cache with "ungenerated_token_id" tokens
        # cache size = self.max_delay + 2
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )

        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        return _LMGenState(cache, initial, graphed_main, graphed_depth)

    # 1/
    # This is only for api/inference generaton mode (hence the torch.no_grad)
    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        assert S == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        # CACHE/ this is the cache size (max_delay + 2)
        CT = state.cache.shape[2]

        # CACHE/ Here the state.cache is written
        # Here we write the input tokens to the cache (respecting the delays)
        
        # STREAM/ the cache logic could be abstracted away. It feels we shouldn't care how to write and read to the cache
        # e.g. we should have only a write_cache and read_cache functions
        # ok, in some sense it's a bit specific, e.g. where to write input/output tokens
        for q_other in range(input_tokens.shape[1]):  # loops over codebooks from input stream
            # For each input codebook:
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]  # [0, 1, 1, 1, 1, 1, 1, 1],  0 for the semantic codebook    
            # get write pos: (offset + delay) % CT
            # offset is the current position in the cache (it's just incremented)
            write_position = (state.offset + delay) % CT
            # k starts after the external stream codebooks
            state.cache[:, k, write_position : write_position + 1] = input_tokens[
                :, q_other
            ]

        # CACHE/ For the very initial tokens (offset <= delay), we write initial values to the cache
        # We could do this we gather similar to the output tokens (see below), right!?
        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            # Only for the very beginning, we extend the initial token for the acoustic
            # token that are delayed, and thus have no good value to take.
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]  # initial has shape [B, K, 1]
        # CACHE/ This is the final input to the model, it's taken from the cache
        # Shape is [1, K + 1 + K, 1] --> So it both stores the input and the outputs stream
        
        # 1/2 This is the final input to the model. It is conposed of the input tokens and the generated tokens 
        # CACHE/ this is just the complete state vector for the current pos
        input_ = state.cache[:, :, position : position + 1]
        

        # CHECK: what below exactly does
        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        # 2/ forward_text
        # This actually calls self.lm_model.forward_text
        # - this applies the "temporal transformer" and returns a combined representation of text and audio tokens
        # - the text_logits are a linear transformation of this representation
        transformer_out, text_logits = state.graphed_main(input_)
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1 
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]
        
        # 3/ depformer_step
        # This actually calls depformer_step
        # Which calls self.lm_model.forward_depformer sequentially for each codebook
        # - transformer_out: output of the temporal transformer
        audio_tokens, audio_logits = state.graphed_depth(text_token, transformer_out)

        # /CACHE: Now we write the generated output back to the cache (i.e. the first 8 tokens in the cache)
        # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
        # Why do we increment the offset by one and don't use delays? --> See below
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1 : lm_model.dep_q + 1, position] = audio_tokens

        # Wait until we have enough tokens to generate an output
        if state.offset <= self.max_delay:
            return None
        # Now we actually apply the delays to the output tokens
        B = state.cache.shape[0]
        # We go actually back in the position to get the output tokens
        # /CACHE: this seems a bit complex: why don't we store tokens at the right position in the cache?
        # This seems to be due to the generation. I.e. the tokens generated now were actually the inputs one time step
        # earlier. --> For training (where all the tokens already exist) we probably don't need this
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)
        input_sequence = input_
        return out, input_sequence, transformer_out, text_logits, text_token, audio_logits, audio_tokens
    # 3/0
    # This is the generate function for the depformer
    # transformer_out: output of the temporal transformer: is not changed
    # depformer_tokens: list of tokens generated sequentially by the depformer
    # CHECK: it seems this function is never called in the codebase
    # -> A: it's actually called in the main `step` function via `graphed_depth`
    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        (B,) = text_token.shape
        # first token set to the text token
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        logits_all = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            # We run it sequentially for all token dimensions
            # The first dim refers to the text token
            # The following dims refer to the audio tokens
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                # 4/ forward_depformer called on prev_token and transformer_out (that stays constant)
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                logits_all.append(logits.clone())
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        logits_all = torch.stack(logits_all, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out, logits_all

class LMNoStream:

    def __init__(self, lm_model: LMModel, check: bool = False):
        self.lm_model = lm_model
        self.check = check

    def forward(self, input_tokens: torch.Tensor, text_tokens: torch.Tensor, output_tokens: torch.Tensor) -> torch.Tensor | None:
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        input_ = torch.cat([output_tokens, text_tokens, input_tokens], dim=1)

        if self.check:
            assert (input_[:, lm_model.audio_offset:] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        # 2/ forward_text
        transformer_out, text_logits = self.lm_model.forward_text(input_)

        assert text_tokens.dim() == 3, text_tokens.shape
        assert text_tokens.shape[2] == 1
        assert text_tokens.shape[1] == 1, "Only one text stream supported."
        text_tokens = text_tokens[:, 0, 0]  # shape is [B]

        # 3/ depformer_step
        audio_tokens = self.depformer_step(text_tokens, transformer_out)
        return audio_tokens

    # def depformer_forward(
    #     self,
    #     input_tokens: torch.Tensor,
    #     text_tokens: torch.Tensor,
    #     output_tokens: torch.Tensor,
    #     transformer_out: torch.Tensor,
    # ) -> torch.Tensor:
    #     for cb_index

    # def depformer_step(
    #     self,
    #     text_token: torch.Tensor,
    #     transformer_out: torch.Tensor,
    # ) -> torch.Tensor:
    #     (B,) = text_token.shape
    #     # first token set to the text token
    #     prev_token = text_token
    #     lm_model = self.lm_model
    #     depformer_tokens: list[torch.Tensor] = []
    #     # assert not lm_model.depformer.is_streaming
    #     # with lm_model.depformer.streaming(B):

    #     for cb_index in range(lm_model.):
    #         input_ = prev_token[:, None, None]
    #         logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
    #         next_token = sample_token(
    #             logits.float(),
    #             self.use_sampling,
    #             self.temp,
    #             self.top_k,
    #         )
    #         assert next_token.shape == (B, 1, 1)
    #         next_token = next_token[:, 0, 0]  # shape is B
    #         depformer_tokens.append(next_token)
    #         prev_token = next_token

    #     assert len(depformer_tokens) == lm_model.dep_q, (
    #         len(depformer_tokens),
    #         lm_model.dep_q,
    #     )
    #     out = torch.stack(depformer_tokens, dim=1)
    #     assert out.shape == (B, lm_model.dep_q), out.shape
    #     return out
