# Edge Impulse - OpenMV Image Classification Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import time, ml, uos, gc, image
from ulab import numpy as np

net = None
labels = None
NUM_INFERENCES = 50

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading

    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
    #net = ml.Model("trained.tflite", load_to_fb=False)
    print("-> Model loaded succesfully!")
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
    print("-> Labels loaded succesfully!")
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    img = image.Image("test_image_96.bmp", copy_to_fb=uos.stat('test_image_96.bmp')[6] > (gc.mem_free() - (64*1024)))
    print("-> Image loaded succesfully!")
except Exception as e:
    raise Exception(f'Failed to load the image. Error:{e}')


try:
    print(f"-> Esecuzione di {NUM_INFERENCES} inferenze di benchmark...")
    times_array = np.zeros(NUM_INFERENCES)

    for i in range(NUM_INFERENCES):
        start_time = time.tick_us()
        net.predict([img])
        end_time = time.tick_us()
        times_array[i] = time.ticks_diff(end_time, start_time) / 1000.0

    print("-> Inferenze completate.")
    avg_time = np.mean(times_array)
    std_dev = np.std(times_array)
    total_time = np.sum(times_array)

    benchmark_metrics = {
        'total_time_ms': total_time,
        'average_time_ms': avg_time,
        'std_dev_ms': std_dev
    }
except Exception as e:
    print(f"Errore durante l'esecuzione delle inferenze: {e}")

if benchmark_metrics:
    print("\n--- RISULTATI DEL BENCHMARK ---")
    print(f"Tempo medio di inferenza: {benchmark_metrics['average_time_ms']:.2f} ms")
    print(f"Deviazione Standard:       {benchmark_metrics['std_dev_ms']:.2f} ms")
    print(f"Tempo totale ({NUM_INFERENCES} esecuzioni): {benchmark_metrics['total_time_ms']:.0f} ms")
    print("---------------------------------")
else:
    print("\nBenchmark non completato a causa di un errore precedente.")




'''
clock = time.clock()
while(True):
    clock.tick()

    #img = sensor.snapshot()

    predictions_list = list(zip(labels, net.predict([img])[0].flatten().tolist()))

    for i in range(len(predictions_list)):
        print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

    print(clock.fps(), "fps")
'''
