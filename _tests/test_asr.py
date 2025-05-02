import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asr import remote_ASRmodel, parseArgs


if __name__ == "__main__":
    args = parseArgs()
    args.token = "oFgLtPOv09NABtwmObu_TZVquKkw-kHEsP-XMv68sxw=|1746808145|unyfv@student.kit.edu"

    asr = remote_ASRmodel(args)
    from threading import Thread
    Thread(target=asr.run_session, daemon=True).start()

    for i in range(3):
        input("Press Enter to indicate you're done:")

        collected = []

        while not asr.text_queue.empty():
            sentence = asr.text_queue.get()
            collected.append(sentence)

        final_text = " ".join(collected).strip()

        if final_text:
            print("Content recognized:", final_text)
        else:
            print("No content recognized.")