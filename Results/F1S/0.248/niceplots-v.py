import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd

# Plot-Einstellungen
plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.get_cmap("tab10").colors

# Hilfsfunktion zum Fit-Plot
def plot_fit(fit_df, color, alpha=0.3):
    emass = fit_df['EffectiveMass']
    error = fit_df['Error']
    reg_low = int(fit_df['RegLow'])
    reg_up = int(fit_df['RegUp']) + 1
    plt.plot([-1, 32], [emass]*2, color=color, lw=1)
    plt.fill_between(range(47)[reg_low:reg_up], emass + error, emass - error, color=color, alpha=alpha)

# Hauptplot
plt.figure(figsize=(7.5, 5.5))
plt.xlabel('Time', fontsize=17)
plt.ylabel(r'$\widetilde{V}$', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)

for i in range(1, 6):
    color = colors[i-1]
    color_light = to_rgba(color, alpha=0.4)

    nsq = pd.read_csv(f'Ratios/V/V-nsq{i}.txt', sep=' ', header=None)
    nsq_disp = pd.read_csv(f'Ratios/V/V-nsq{i}-Disp.txt', sep=' ', header=None)
    fit = pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit.csv', sep='\s')
    fit_disp = pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit-Disp.csv', sep='\s')

    plt.errorbar(range(29), nsq[0][1:30], yerr=nsq[1][1:30], fmt='o', color=color, markersize=4, label=fr'$n^2={i}$')
    plt.errorbar(range(29), nsq_disp[0][1:30], yerr=nsq_disp[1][1:30], fmt='s', color=color_light, markersize=4, label=fr'$n^2={i}$ Disp')

    plot_fit(fit, color)
    plot_fit(fit_disp, color_light)

plt.title(r'Effective Mass $\widetilde{V}$ inkl. Dispersion', fontsize=18)
plt.annotate(r'\textbf{preliminary}', xy=(0.17, 0.03), xycoords='axes fraction', fontsize=14, color='grey', alpha=0.7)
plt.legend(fontsize=11, loc='lower right', frameon=False, ncol=2)
plt.tight_layout()
plt.savefig('Niceplot-V.pdf', transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# Einzelplots je n^2
for i in range(1, 6):
    color = colors[i-1]
    color_light = to_rgba(color, alpha=0.4)

    nsq = pd.read_csv(f'Ratios/V/V-nsq{i}.txt', sep=' ', header=None)
    nsq_disp = pd.read_csv(f'Ratios/V/V-nsq{i}-Disp.txt', sep=' ', header=None)
    fit = pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit.csv', sep='\s')
    fit_disp = pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit-Disp.csv', sep='\s')

    plt.figure(figsize=(6.5, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel(r'$\widetilde{V}$', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)

    plt.errorbar(range(29), nsq[0][1:30], yerr=nsq[1][1:30], fmt='o', color=color, markersize=4, label='Original')
    plt.errorbar(range(29), nsq_disp[0][1:30], yerr=nsq_disp[1][1:30], fmt='s', color=color_light, markersize=4, label='Disp')

    plot_fit(fit, color)
    plot_fit(fit_disp, color_light)

    plt.legend(fontsize=11, loc='lower right', frameon=False)
    plt.title(rf'Effective Mass $\widetilde{{V}}$, $n^2={i}$', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'V-nsq{i}-Comparison.pdf', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()
