from colorthief import ColorThief
import os
import csv
import time

start = time.time()

colunas = ['image', 'era']

for i in range(50):
	coluna = 'c' + str(i+1)
	colunas.append(coluna)

# print(colunas)
conteudo = [colunas]

for root, dirs, files in os.walk("/home/mariana/percepcao", topdown=False):
	for name in files:
		imagem = name
		pasta = root.split("/")[-1]
		arquivo = os.path.join(root, name)
		paleta = []
		tudo = [imagem, pasta]
		if pasta != 'percepcao':
			ct = ColorThief(arquivo)
			num_cores = 50
			paleta = ct.get_palette(color_count=num_cores)
			while len(paleta) < 50:
				num_cores += 1
				paleta = ct.get_palette(color_count=num_cores)
			for c in paleta:
				tudo.append(c)
			conteudo.append(tudo)


# print(conteudo)

with open("colors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(conteudo)

end = time.time()
print(end - start)