import pandas as pd
import matplotlib.pyplot as plt

ens= 'C2'  
ens_ow='010'
m='0.300'

# === Load data1 ===
df1 = pd.read_csv("../Results/Crosschecks/AB/Crosscheck-{}-{}.csv".format(ens,m), sep="\t", header=None, names=["value", "error", "pval"])
#df1 = df1.iloc[0:15].reset_index(drop=True)
df1["id"] = df1.index
#print(df1)
# === Load data2 ===
df2 = pd.read_csv("../Results/Crosschecks/OW/pdf/{}-amc{}.txt".format(ens_ow,m), sep="\s+", header=None, names=["index", "value", "error", "other1", "other2"])

# Map indices from your list
#map_indices = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
map_indices = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22,23,24,25,26,27]
df2_selected = df2.loc[map_indices].reset_index(drop=True)
df2_selected["id"] = df2_selected.index

observable_names = [r"$E_{Eff}(B_s,n^2=0)$", r"$E_{Eff}(D_s^*,n^2=0)$",r"$E_{Eff}(D_s^*,n^2=1)$",r"$E_{Eff}(D_s^*,n^2=2)$",r"$E_{Eff}(D_s^*,n^2=3)$",r"$E_{Eff}(D_s^*,n^2=4)$", r"$E_{Eff}(D_s^*,n^2=5)$", r'$V(n^2=1)$',r'$V(n^2=2)$',r'$V(n^2=3)$',r'$V(n^2=4)$',r'$V(n^2=5)$', r'$A_0(n^2=1)$',r'$A_0(n^2=2)$',r'$A_0(n^2=3)$',r'$A_0(n^2=4)$',r'$A_0(n^2=5)$',r'$A_1(n^2=0)$',r'$A_1(n^2=1)$',r'$A_1(n^2=2)$',r'$A_1(n^2=4)$',r'$A_1(n^2=5)$']

#print(df2_selected)

# === Plotting Function ===
def plot_range(start, end, df1, df2, title,labels=None,ylim=None):
    ids = df1['id'].iloc[start+1:end+2]
    x_labels = labels[start:end+1] if labels else ids

    plt.figure(figsize=(10, 5))
    print(df1['value'].iloc[start+1:end+2])
    plt.errorbar(ids, df1['value'].iloc[start+1:end+2], yerr=df1['error'].iloc[start+1:end+2],
                 fmt='o', label='AB', capsize=3, color='blue')
    print(df2['value'].iloc[start:end+1])
    plt.errorbar(ids, df2['value'].iloc[start:end+1], yerr=df2['error'].iloc[start:end+1],
                 fmt='o', label='OW', capsize=3, color='red',marker='x')

    if ylim is not None:
        plt.ylim(ylim)
    #plt.xlabel("Observable ID")
    #plt.ylabel("Value")
    plt.xticks(ticks=ids, labels=x_labels, rotation=45)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../Results//Crosschecks/Comparison-{}-{}_{}-{}.png'.format(start, end,ens,m), dpi=300)
    plt.close()

df1["value"] = pd.to_numeric(df1["value"], errors="coerce")
df1["error"] = pd.to_numeric(df1["error"], errors="coerce")
df2_selected["value"] = pd.to_numeric(df2_selected["value"], errors="coerce")
df2_selected["error"] = pd.to_numeric(df2_selected["error"], errors="coerce")

# === Plot in two parts ===
#ylim=(0.8, 2.3)
plot_range(0, 6, df1, df2_selected, "",labels=observable_names)
plot_range(7, 21, df1, df2_selected, "",labels=observable_names)
