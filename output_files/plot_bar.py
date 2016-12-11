import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_stacked(dfall, labels=None, title="MFCC Features Generation Time Profile on CPU vs GPU",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.savefig("myfig.png")
    return axe

# create fake dataframes

df1_data = np.ndarray((5,6), buffer = np.array([345215,1363598,198498,100687,109318,759472,
395468,1562099,227394,115345,125231,870030,
380575,1503270,218830,111001,120516,837264,
415195,1640022,238737,121099,131478,913430,
440575,1740270,253330,128501,139515,969265]), dtype = long)

# df2_data_arr = np.array([124148,552457,74489,33106,39313,231742,
# 204681,910826,122808,54582,64816,382070,
# 200948,894220,120569,53586,63634,375103,
# 257684,1146693,154610,68715,81600,481009,
# 304757,1356170,182854,81269,96507,568880])

df2_data = np.ndarray((5,6), buffer = np.array([124148,552457,74489,33106,39313,231742,
204681,910826,122808,54582,64816,382070,
200948,894220,120569,53586,63634,375103,
257684,1146693,154610,68715,81600,481009,
304757,1356170,182854,81269,96507,568880]), dtype = long)
df1 = pd.DataFrame(df1_data,
                   index=["5s", "10s", "15s", "20s", "25s"],
                   columns=["Segmenter", "FFT", "Mel Filterbank", "DCT", "Delta", "Normalize"])
df2 = pd.DataFrame(df2_data,
                   index=["5s", "10s", "15s", "20s", "25s"],
                   columns=["Segmenter", "FFT", "Mel Filterbank", "DCT", "Delta", "Normalize"])


# Then, just call :
plot_clustered_stacked([df1, df2],["CPU", "GPU"])