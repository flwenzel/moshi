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
