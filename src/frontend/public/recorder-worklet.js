class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input.length > 0) {
      const mono = input[0];
      this.port.postMessage(mono);
    }
    return true;
  }
}
registerProcessor('recorder-processor', RecorderProcessor);
