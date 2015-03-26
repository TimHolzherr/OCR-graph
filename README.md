# OCR-graph
Optical character recognition with the help of a SVM and a probabilistic graphical model, implemented in Python

The purpose of this project is to learn more about PGM's.
The setting is somewhat aritifical: A SVM is trained with a small dataset for the OCR task, the task is to improve the occuracy by making use of the external knowledge (in this case a dictionary).
The idea is to implemend the algorithems in PGM in the general case and use the OCR problem as a test case.
This OCR problem is originaly from Daphne Koller in ther Stanford Course about PGMs, all the algorithems are implemended based on the ideas of her course.
On of the motivation of this projectc was to implemend the assigments of Daphne's course on my one from scratch in "nice" python code (instead of Matlab)  