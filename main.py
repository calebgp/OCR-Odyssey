# import cv2
# import pytesseract
# import numpy as np
# from skimage.segmentation import clear_border
# import re

# imagem = cv2.imread("placa1.jpg")

# text = pytesseract.image_to_string(imagem)
# print(text)

# # cv2.imshow("Image", imagem)
# # cv2.waitKey(0)

# imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# kernel_retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 23))
# chapeu_preto = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, kernel_retangular)

# sobel_x = cv2.Sobel(chapeu_preto, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
# sobel_x = np.absolute(sobel_x)
# sobel_x = sobel_x.astype("uint8")

# sobel_x = cv2.GaussianBlur(sobel_x, (5, 5), 0)
# sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, kernel_retangular)

# valor, limiarizacao = cv2.threshold(
#     sobel_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
# )

# kernel_quadrado = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# limiarizacao = cv2.erode(limiarizacao, kernel_quadrado, iterations=2)
# limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations=2)

# fechamento = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel_quadrado)
# valor, mascara = cv2.threshold(fechamento, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# limiarizacao = cv2.bitwise_and(limiarizacao, limiarizacao, mask=mascara)
# limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations=2)
# limiarizacao = cv2.erode(limiarizacao, kernel_quadrado)

# limiarizacao = clear_border(limiarizacao)

# contornos, hierarquia = cv2.findContours(
#     limiarizacao, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
# )
# contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

# for contorno in contornos:
#     x, y, w, h = cv2.boundingRect(contorno)
#     proporcao = float(w) / h
#     print(proporcao)
#     if proporcao >= 4.5 and proporcao <= 5:
#         placa = imagem[y : y + h, x : x + w]

#         limiar = 1
#         valor, lim_simples = cv2.threshold(placa, limiar, 255, cv2.THRESH_BINARY)
#         regiao_interesse = clear_border(lim_simples)
#         cv2.imshow("Placa", placa)
#         cv2.waitKey(0)

#         cv2.imshow("Região de Interesse", regiao_interesse)
#         cv2.waitKey(0)

#         config_tesseract = "--tessdata-dir tessdata --psm 6"
#         texto = pytesseract.image_to_string(
#             regiao_interesse, lang="por", config=config_tesseract
#         )
#         print("." + texto)
#         texto_extraido = re.search("\w{3}\d{1}\w{1}\d{2}", texto)
#         print(texto_extraido)
import cv2
import pytesseract
import numpy as np
from skimage.segmentation import clear_border
import re

imagem = cv2.imread("placa1.jpg")


imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

kernel_retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 13))
chapeu_preto = cv2.morphologyEx(imagem_gray, cv2.MORPH_BLACKHAT, kernel_retangular)

sobel_x = cv2.Sobel(chapeu_preto, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
sobel_x = np.absolute(sobel_x)
sobel_x = sobel_x.astype("uint8")

sobel_x = cv2.GaussianBlur(sobel_x, (5, 5), 0)
sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, kernel_retangular)

_, limiarizacao = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel_quadrado = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
limiarizacao = cv2.erode(limiarizacao, kernel_quadrado, iterations=2)
limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations=2)

fechamento = cv2.morphologyEx(imagem_gray, cv2.MORPH_CLOSE, kernel_quadrado)
_, mascara = cv2.threshold(fechamento, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

limiarizacao = cv2.bitwise_and(limiarizacao, limiarizacao, mask=mascara)
limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations=2)
limiarizacao = cv2.erode(limiarizacao, kernel_quadrado)

limiarizacao = clear_border(limiarizacao)

contornos, hierarquia = cv2.findContours(
    limiarizacao, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]
bordas = cv2.Canny(imagem_gray, 100, 300)
cv2.imshow("Canny", bordas)
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    proporcao = float(w) / h
    print(proporcao)
    if proporcao >= 4 and proporcao <= 5:
        placa = imagem[y : y + h, x : x + w]
        placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
        valor, regiao_interesse = cv2.threshold(
            placa_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        # regiao_interesse = cv2.morphologyEx(
        #     regiao_interesse,
        #     cv2.MORPH_GRADIENT,
        #     kernel_quadrado,
        # )
        regiao_interesse = cv2.morphologyEx(
            regiao_interesse,
            cv2.MORPH_BLACKHAT,
            kernel_retangular,
        )
        # regiao_interesse = cv2.erode(regiao_interesse, kernel_retangular, iterations=2)
        # regiao_interesse = cv2.dilate(regiao_interesse, kernel_retangular, iterations=2)
        regiao_interesse = clear_border(regiao_interesse)
        cv2.imshow("Placa", placa_gray)
        cv2.waitKey(0)
        cv2.imshow("Região de Interesse", regiao_interesse)

        config_tesseract = "--tessdata-dir tessdata "
        texto = pytesseract.image_to_string(
            regiao_interesse, lang="por", config=config_tesseract
        )
        print(texto)
        texto_extraido = re.search("\w{3}\d{1}\w{1}\d{2}", texto)
        print(texto_extraido)
