# How does the stream module work?

## `StreamingModule`
* Base module
* Each streaming component has a streaming state, which is just a dict[str, Tensor]
  *  By convention, the first dim of each tensor must be the batch size.
  *   If `self._is_streaming` is True, the component should use and remember
    the proper state inside `self._streaming_state`.
    * *Question:* what does remember mean here?

`_start_streaming`
* Defines internal `_start_streaming` and passes this to `_apply_named_streaming`
  * for a given module, this just assigns an init value module._streaming_state
  * `_init_streaming_state` has to be implemented by the module
* `_apply_named_streaming`: 

`_apply_named_streaming
* Used to apply a function to the module and all its children


## `LMGen(StreamingModule[_LMGenState]`
* Check in detail how the token cache would work in training compared to generation
  * Probably the "going back in time" logic is not needed then
  * It's only needed for streaming since at time t you need to output the current state and not wait `max_delay` steps more to do it
* Also compare this to the pattern provider implementation in MusicGen

## StreamingMultiheadAttention(StreamingModule[_MHAState])


## Open questions
* What is the CudaGRAPH stuff and why is it part of the streaming state?