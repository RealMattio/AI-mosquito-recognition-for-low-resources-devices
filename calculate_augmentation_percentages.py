from math import ceil

starting_number = 4449
total_number_with_augmentation = 26694
percentages = [0, 25, 50, 75, 100]
imm_gen = total_number_with_augmentation - starting_number
print(f"Numero totale di immagini con augmentation: {total_number_with_augmentation}")


for percentage in percentages:
    x = int(imm_gen * percentage / 100)
    y = ceil((x + starting_number) / starting_number * 100)
    print(f"{percentage}% - Percentuale rispetto al dataset originale: {y:.2f}% - Numero di immagini con augmentation: {x}")