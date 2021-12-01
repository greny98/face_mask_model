import time

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter("tflite_model/model.tflite")
    interpreter.allocate_tensors()
    img = np.ones(shape=(1, 320, 320, 3), dtype=np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, img)

    # Run the inference
    interpreter.invoke()
    # output_details = interpreter.get_output_details()[0]

    start = time.time()
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    print(time.time() - start)
