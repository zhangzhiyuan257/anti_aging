f = open("nega_0.9.fasta","r")
h = open("positive_0.9.fasta","r")
data_n = f.read()
data_p = h.read()

data_n_list = data_n.split("\n")
data_p_list = data_p.split("\n")

data_n_l = [i for i in data_n_list if ">AA" not in i]
data_p_l = [i for i in data_p_list if ">AA" not in i]

string_p, string_n = "",""
for i in data_n_l:
    string_n += i
    
for i in data_p_l:
    string_p += i


from collections import Counter

p = Counter(string_p)
for i in p:
    p[i] = p[i]/len(string_p)
n = Counter(string_n)
for i in n:
    n[i] = n[i]/len(string_n)


import matplotlib.pyplot as plt
import numpy as np

species = ["A", "R","N","D","C","Q","E","G","H","I","L","K",
           "M","F","P","S","T","W","Y","V"]

Antiaging  = [round(p[i],4) for i in species]
Non_antiaging = [round(n[i],4) for i in species] 
    
penguin_means = {
    "Antiaging  peptide": Antiaging,
    "Non antiaging peptide": Non_antiaging,
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3,rotation=70)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage')
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species)
#ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 0.12)
plt.show()
