# CatsAndDogs

## Аугментации
Для этой задачи на тренировочном наборе данных выполнялись следующие аугментации:
 
1. VerticalFlip, HorizontalFlip и Rotate: Это может быть полезно, поскольку кошки и собаки могут быть сфотографированы с разных ракурсов и углов. Кошка вверх ногами всеравно кошка.
2. CLAHE (Contrast Limited Adaptive Histogram Equalization): Эта аугментация улучшает контраст изображения, тем самым лучше распознаются объекты на изображении.
3. RandomBrightnessContrast: Случайным образом изменяет яркость и контраст изображения. Позволяет модели работать с данными при различных условиях освещения.
4. RandomGamma: Изменяет гамму изображения, что также может помочь модели лучше справляться с различными условиями освещения.
5. Resize: Меняет размер изображения. (в данном случае, 224x224). Это необходимо, поскольку модель имеет фиксированый входной слой.
6. Normalize: Нормализует значения пикселей изображения.

Для тестового и валидационных наборов данных выполнялись только 5. и 6.
