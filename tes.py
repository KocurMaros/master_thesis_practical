import numpy as np
# Nová confusion matrix – absolútne počty
cm_absolute = np.array([
    [96, 7, 11, 20, 3, 14, 11],
    [11, 58, 0, 23, 32, 29, 7],
    [2, 0, 39, 8, 3, 11, 11],
    [5, 10, 4, 1107, 31, 18, 10],
    [4, 12, 0, 64, 482, 79, 39],
    [3, 9, 0, 64, 40, 351, 11],
    [2, 0, 16, 10, 20, 6, 275]
])

# Výpočet presnosti
correct_absolute = np.trace(cm_absolute)
total_absolute = np.sum(cm_absolute)
accuracy_absolute = correct_absolute / total_absolute

print (accuracy_absolute * 100)  # v percentách