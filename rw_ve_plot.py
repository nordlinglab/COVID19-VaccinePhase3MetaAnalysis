import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

##############################
# Recalculate vaccine efficacy
##############################


def plot_compare_reproduced_vaccine_efficacy_scalar_plot(paper_vaccine_efficacy, paper_lower_bound,
                                                         paper_upper_bound, paper_ve_methods, paper_ci_methods,
                                                         RW_vaccine_efficacy, RW_lower_bound, RW_upper_bound, save_fig=True):

    plt.plot()
    for i in range(len(paper_vaccine_efficacy)):
        if paper_lower_bound[i] == 0 and paper_upper_bound[i] == 0:
            x_error = np.array([[0, 0]]).T
        else:
            x_error = np.array([[paper_vaccine_efficacy[i]-paper_lower_bound[i],
                                paper_upper_bound[i]-paper_vaccine_efficacy[i]]]).T

        x_error[x_error < 0] = 0

        if RW_lower_bound[i] == 0 and RW_upper_bound[i] == 0:
            y_error = np.array([[0, 0]]).T
        else:
            y_error = np.array([[RW_vaccine_efficacy[i]-RW_lower_bound[i],
                                RW_upper_bound[i]-RW_vaccine_efficacy[i]]]).T

        plt.errorbar(paper_vaccine_efficacy[i], RW_vaccine_efficacy[i],
                     xerr=x_error, yerr=y_error, fmt='.', color='k', elinewidth=0.5, capsize=0)
    plt.plot([0, 105], [0, 105], '--k')
    plt.xlabel(
        'Reported vaccine efficacy with $95\%$ CIs in the original study')
    plt.ylabel(
        'Vaccine efficacy with $95\%$ CIs based on relative risk and \n Poisson regression with robust error')
    plt.grid()
    plt.xlim(0, 105)
    plt.ylim(0, 105)

    if save_fig == True:
        plt.savefig('RW_vaccine_efficacy_reproduction.pdf')


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), numpoints=1)


def plot_compare_reproduced_vaccine_efficacy_error_bar_plot(paper_vaccine_efficacy, paper_lower_bound,
                                                            paper_upper_bound, paper_ve_methods, paper_ci_methods,
                                                            RW_vaccine_efficacy, RW_lower_bound, RW_upper_bound,
                                                            references, save_fig=True):
    # Initialization
    palette_all = sns.color_palette(
        'Set1', n_colors=len(paper_vaccine_efficacy)+2)
    # Remove yellow
    palette = palette_all[0:5] + palette_all[6:14]
    for i in range(len(palette_all)//len(palette)):
        palette = palette + palette[0:13]
    palette = palette[0:len(paper_vaccine_efficacy)]

    # Sort
    sort_index = np.argsort(paper_vaccine_efficacy)
    # sort_index = sort_index[::-1]
    paper_vaccine_efficacy = paper_vaccine_efficacy[sort_index]
    paper_lower_bound = paper_lower_bound[sort_index]
    paper_upper_bound = paper_upper_bound[sort_index]
    paper_ve_methods = paper_ve_methods[sort_index]
    paper_ci_methods = paper_ci_methods[sort_index]
    RW_vaccine_efficacy = RW_vaccine_efficacy[sort_index]
    RW_lower_bound = RW_lower_bound[sort_index]
    RW_upper_bound = RW_upper_bound[sort_index]
    references = references[sort_index]

    # Markers and colors
    all_markers = np.array(
        ['o', 'v', '^', '<', '>', 's', 'P', 'p', '*', 'h', 'H', 'X', 'D'])
    ve_method_markers_dict = dict(
        zip(np.unique(paper_ve_methods), all_markers[0:len(np.unique(paper_ve_methods))]))
    ve_method_color_dict = dict(
        zip(np.unique(paper_ve_methods), palette[0:len(np.unique(paper_ve_methods))]))

    # Confidence interval and color
    ci_method_color_dict = dict(
        zip(np.unique(paper_ci_methods), palette[0:len(np.unique(paper_ci_methods))]))

    # Plot
    index = 0
    index_array = np.array([])
    fig = plt.figure(figsize=(12, len(paper_vaccine_efficacy)*0.4))
    ax = fig.subplots(1, 1)
    for i in range(len(sort_index)):
        efficacy = np.array(
            [paper_vaccine_efficacy[i], RW_vaccine_efficacy[i]])
        lower_bounds = np.array([paper_lower_bound[i], RW_lower_bound[i]])
        upper_bounds = np.array([paper_upper_bound[i], RW_upper_bound[i]])
        paper_ve_method = paper_ve_methods[i]
        paper_ci_method = paper_ci_methods[i]

        if lower_bounds[0] == 0 and upper_bounds[0] == 0:
            xerr0 = np.array([[0, 0]]).T
        else:
            xerr0 = np.array([[paper_vaccine_efficacy[i]-paper_lower_bound[i],
                               paper_upper_bound[i]-paper_vaccine_efficacy[i]]]).T

        xerr0[xerr0 < 0] = 0
        if lower_bounds[1] == 0 and upper_bounds[1] == 0:
            xerr1 = np.array([[0, 0]]).T
        else:
            xerr1 = np.array([[RW_vaccine_efficacy[i]-RW_lower_bound[i],
                               RW_upper_bound[i]-RW_vaccine_efficacy[i]]]).T

        line1 = ax.errorbar(efficacy[0], index, xerr=xerr0,
                            fmt=ve_method_markers_dict[paper_ve_method],
                            ms=5, ecolor=ci_method_color_dict[paper_ci_method],
                            markerfacecolor=ve_method_color_dict[paper_ve_method],
                            label=paper_ve_method+'\n'+paper_ci_method)
        line2 = ax.errorbar(efficacy[1], index+1, xerr=xerr1,
                            fmt=ve_method_markers_dict['Relative risk'],
                            ms=5,
                            ecolor=ci_method_color_dict['Poisson regression with robust error variance'],
                            markerfacecolor=ve_method_color_dict['Relative risk'])
        index += 4
        index_array = np.append(index_array, index)
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    # handles = [h[0] for h in handles]
    # use them in the legend
    # ax.legend(handles[0:2], labels[0:2], loc='upper right', numpoints=1)
    legend_without_duplicate_labels(ax)
    # ax.legend(loc='upper right', numpoints=1)

    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(index_array-4, references, fontsize=20)
    plt.ylim([-1, index])
    plt.xlim([0, 200])
    plt.gca().invert_yaxis()
    # plt.gca().yaxis.grid()
    plt.gca().xaxis.grid()
    plt.plot([30, 30], [-1, index], 'k--')
    plt.plot([50, 50], [-1, index], 'k--')
    plt.xlabel('Vaccine efficacy with $95\%$ CIs', fontsize=20)

    if save_fig == True:
        plt.savefig('RW_vaccine_efficacy_reproduction_error_bar.pdf')


def different_between_efficacy(paper_vaccine_efficacy, paper_ve_methods, RW_vaccine_efficacy, save_fig=True):
    # Initialization
    palette = sns.color_palette(
        'Set1', n_colors=len(np.unique(paper_ve_methods)))

    # Remove outliers
    efficacy_remove_map = RW_vaccine_efficacy != 100
    remove_map = efficacy_remove_map
    paper_vaccine_efficacy = paper_vaccine_efficacy[remove_map]
    paper_ve_methods = paper_ve_methods[remove_map]
    RW_vaccine_efficacy = RW_vaccine_efficacy[remove_map]

    # Efficacy
    fig = plt.figure(figsize=(8, 5))
    unique_paper_ve_methods = np.unique(paper_ve_methods)
    efficacy_difference_dict = {}
    for i, paper_ve_method in enumerate(unique_paper_ve_methods):
        index_map = paper_ve_methods == paper_ve_method
        paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[index_map]
        RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[index_map]
        efficacy_difference = paper_vaccine_efficacy_tmp-RW_vaccine_efficacy_tmp
        efficacy_difference_dict[paper_ve_method] = efficacy_difference

    # 'transpose' items to parallel key, value lists
    labels, data = [*zip(*efficacy_difference_dict.items())]
    
    means = [data[i].mean() for i in range(len(data))]
    medians = [np.median(data[i]) for i in range(len(data))]
    plt.plot(range(0, len(labels)), means, '^', color='black', alpha=1, label='mean', markersize=7)
    plt.plot(range(0, len(labels)), medians, 's', color='black', alpha=1, label='mean', markersize=7)
    print('Means: ', means)
    print('Medians: ', medians)

    sns.violinplot(data, inner=None, cut=0, palette=palette, scale='width')
    sns.swarmplot(data=data, color='black', alpha=0.5, size=4)
    plt.fill_betweenx(y=[-15, 20], x1=-0.5, x2=2.5, color='gray', alpha=0.2)
    plt.xlim(-0.5, 4.5)
    plt.ylim([-15, 18])
    plt.xticks(range(0, len(labels)), labels)
    plt.xticks(rotation=15, ha='right')
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.grid()
    plt.ylabel(
        'Difference between original vaccine efficacy and \n vaccine efficacy estimated by relative risk (%)')
    plt.legend(loc='lower right', numpoints=1)
    if save_fig == True:
        plt.savefig('Difference_efficacy_RW2022.pdf')
    return fig


def distance_between_CI(paper_vaccine_efficacy, paper_lower_bound,
                        paper_upper_bound, paper_ve_methods, paper_ci_methods,
                        RW_vaccine_efficacy, RW_lower_bound, RW_upper_bound,
                        save_fig=True):
    # Initialization
    palette_all = sns.color_palette(
        'Set1', n_colors=len(paper_vaccine_efficacy)+2)
    # Remove yellow
    palette = palette_all[0:5] + palette_all[6:14]
    for i in range(len(palette_all)//len(palette)):
        palette = palette + palette[0:13]
    palette = palette[0:len(paper_vaccine_efficacy)]
    # Confidence interval and color
    ci_method_color_dict = dict(
        zip(np.unique(paper_ci_methods), palette[0:len(np.unique(paper_ci_methods))]))

    # Sort
    sort_index = np.argsort(paper_ci_methods)
    paper_vaccine_efficacy = paper_vaccine_efficacy[sort_index]
    paper_lower_bound = paper_lower_bound[sort_index]
    paper_upper_bound = paper_upper_bound[sort_index]
    paper_ve_methods = paper_ve_methods[sort_index]
    paper_ci_methods = paper_ci_methods[sort_index]
    RW_vaccine_efficacy = RW_vaccine_efficacy[sort_index]
    RW_lower_bound = RW_lower_bound[sort_index]
    RW_upper_bound = RW_upper_bound[sort_index]

    ###########################
    # Time to event lower bound
    ###########################
    time_to_event = True
    # VE method map that based on time to event involved in the estimation method
    if time_to_event:
        ve_method_list = ['Attack rate',
                          'Hazard ratio', 'Incidence rate ratio']
    else:
        ve_method_list = ['Odds ratio', 'Relative risk']
    ve_method_map = np.isin(paper_ve_methods, ve_method_list)

    # Remove our 100% efficacy data and paper's 0 CI
    efficacy_remove_map = RW_vaccine_efficacy != 100
    ci_remove_map = paper_lower_bound != 0
    remove_map = efficacy_remove_map & ci_remove_map & ve_method_map

    paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[remove_map]
    paper_lower_bound_tmp = paper_lower_bound[remove_map]
    paper_upper_bound_tmp = paper_upper_bound[remove_map]
    paper_ve_methods_tmp = paper_ve_methods[remove_map]
    paper_ci_methods_tmp = paper_ci_methods[remove_map]
    RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[remove_map]
    RW_lower_bound_tmp = RW_lower_bound[remove_map]
    RW_upper_bound_tmp = RW_upper_bound[remove_map]

    # Lower
    fig1 = fig, ax = plt.subplots(figsize=(9, 7))
    paper_lower_lengths = paper_vaccine_efficacy_tmp - paper_lower_bound_tmp
    RW_lower_lengths = RW_vaccine_efficacy_tmp - RW_lower_bound_tmp
    unique_paper_ci_methods = np.unique(paper_ci_methods_tmp)
    lower_difference_dict = {}
    for i, paper_ci_method in enumerate(unique_paper_ci_methods):
        lower_index_map = paper_ci_methods_tmp == paper_ci_method
        paper_lower_lengths_tmp = paper_lower_lengths[lower_index_map]
        RW_lower_lengths_tmp = RW_lower_lengths[lower_index_map]
        lower_difference = paper_lower_lengths_tmp - RW_lower_lengths_tmp
        lower_difference_dict[paper_ci_method] = lower_difference
    shaded_x_range = len(lower_difference_dict)

    ##############################
    # No time to event lower bound
    ##############################
    time_to_event = False
    # VE method map that based on time to event involved in the estimation method
    if time_to_event:
        ve_method_list = ['Attack rate',
                          'Hazard ratio', 'Incidence rate ratio']
    else:
        ve_method_list = ['Odds ratio', 'Relative risk']
    ve_method_map = np.isin(paper_ve_methods, ve_method_list)

    remove_map = efficacy_remove_map & ci_remove_map & ve_method_map
    paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[remove_map]
    paper_lower_bound_tmp = paper_lower_bound[remove_map]
    paper_upper_bound_tmp = paper_upper_bound[remove_map]
    paper_ve_methods_tmp = paper_ve_methods[remove_map]
    paper_ci_methods_tmp = paper_ci_methods[remove_map]
    RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[remove_map]
    RW_lower_bound_tmp = RW_lower_bound[remove_map]
    RW_upper_bound_tmp = RW_upper_bound[remove_map]

    # Lower
    paper_lower_bound_tmp[paper_lower_bound_tmp<0] = 0
    RW_lower_bound_tmp[RW_lower_bound_tmp<0] = 0
    paper_lower_lengths = paper_vaccine_efficacy_tmp - paper_lower_bound_tmp
    RW_lower_lengths = RW_vaccine_efficacy_tmp - RW_lower_bound_tmp
    unique_paper_ci_methods = np.unique(paper_ci_methods_tmp)
    for i, paper_ci_method in enumerate(unique_paper_ci_methods):
        lower_index_map = paper_ci_methods_tmp == paper_ci_method
        paper_lower_lengths_tmp = paper_lower_lengths[lower_index_map]
        RW_lower_lengths_tmp = RW_lower_lengths[lower_index_map]
        lower_difference = paper_lower_lengths_tmp - RW_lower_lengths_tmp
        if paper_ci_method in lower_difference_dict:
            paper_ci_method = paper_ci_method+'*'
        lower_difference_dict[paper_ci_method] = lower_difference

    # 'transpose' items to parallel key, value lists
    labels, data = [*zip(*lower_difference_dict.items())]
    labels = [label.replace('*', '') for label in labels]
    color_palette = [
        ci_method_color_dict[label.strip('*')] for label in labels]
    sns.violinplot(data, inner='quart', cut=0, palette=color_palette, scale='width')
    sns.swarmplot(data=data, color='black', alpha=0.5, size=6)
    plt.xticks(range(0, len(labels)), labels)
    plt.xticks(rotation=30, ha='right')
    plt.grid()
    plt.ylabel('Difference between original lower bound \n and lower bound estimated by \n Poisson regression with robust error variance (%)')
    plt.ylim([-100, 30])
    # if time_to_event:
    #     plt.fill_between(np.arange(-0.5, len(labels)+0.5), -10, 10, color='gray', alpha=0.2)
    plt.xlim(-0.5, len(labels)-0.5)
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.fill_between(np.arange(-0.5, shaded_x_range+0.5), 
                     -100, 30, color='gray', alpha=0.2)
    if save_fig == True:
        plt.savefig(f'Difference_lower_RW2022.pdf')

    ###########################
    # Time to event lower bound
    ###########################
    time_to_event = True
    # VE method map that based on time to event involved in the estimation method
    if time_to_event:
        ve_method_list = ['Attack rate',
                          'Hazard ratio', 'Incidence rate ratio']
    else:
        ve_method_list = ['Odds ratio', 'Relative risk']
    ve_method_map = np.isin(paper_ve_methods, ve_method_list)

    # Remove our 100% efficacy data and paper's 0 CI
    efficacy_remove_map = RW_vaccine_efficacy != 100
    ci_remove_map = paper_lower_bound != 0
    remove_map = efficacy_remove_map & ci_remove_map & ve_method_map

    paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[remove_map]
    paper_lower_bound_tmp = paper_lower_bound[remove_map]
    paper_upper_bound_tmp = paper_upper_bound[remove_map]
    paper_ve_methods_tmp = paper_ve_methods[remove_map]
    paper_ci_methods_tmp = paper_ci_methods[remove_map]
    RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[remove_map]
    RW_lower_bound_tmp = RW_lower_bound[remove_map]
    RW_upper_bound_tmp = RW_upper_bound[remove_map]

    # upper
    fig1 = plt.figure(figsize=(10, 5))
    # paper_upper_bound_tmp[paper_upper_bound_tmp>100] = 100
    # RW_upper_bound_tmp[RW_upper_bound_tmp>100] = 100
    paper_upper_lengths = paper_vaccine_efficacy_tmp - paper_upper_bound_tmp
    RW_upper_lengths = RW_vaccine_efficacy_tmp - RW_upper_bound_tmp
    unique_paper_ci_methods = np.unique(paper_ci_methods_tmp)
    upper_difference_dict = {}
    for i, paper_ci_method in enumerate(unique_paper_ci_methods):
        upper_index_map = paper_ci_methods_tmp == paper_ci_method
        paper_upper_lengths_tmp = paper_upper_lengths[upper_index_map]
        RW_upper_lengths_tmp = RW_upper_lengths[upper_index_map]
        upper_difference = paper_upper_lengths_tmp - RW_upper_lengths_tmp
        # print(i, upper_difference)
        upper_difference_dict[paper_ci_method] = upper_difference
    shaded_x_range = len(upper_difference_dict)

    ##############################
    # No time to event upper bound
    ##############################
    time_to_event = False
    # VE method map that based on time to event involved in the estimation method
    if time_to_event:
        ve_method_list = ['Attack rate',
                          'Hazard ratio', 'Incidence rate ratio']
    else:
        ve_method_list = ['Odds ratio', 'Relative risk']
    ve_method_map = np.isin(paper_ve_methods, ve_method_list)

    remove_map = efficacy_remove_map & ci_remove_map & ve_method_map
    paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[remove_map]
    paper_lower_bound_tmp = paper_lower_bound[remove_map]
    paper_upper_bound_tmp = paper_upper_bound[remove_map]
    paper_ve_methods_tmp = paper_ve_methods[remove_map]
    paper_ci_methods_tmp = paper_ci_methods[remove_map]
    RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[remove_map]
    RW_lower_bound_tmp = RW_lower_bound[remove_map]
    RW_upper_bound_tmp = RW_upper_bound[remove_map]

    # upper
    paper_upper_lengths = paper_vaccine_efficacy_tmp - paper_upper_bound_tmp
    RW_upper_lengths = RW_vaccine_efficacy_tmp - RW_upper_bound_tmp
    unique_paper_ci_methods = np.unique(paper_ci_methods_tmp)
    for i, paper_ci_method in enumerate(unique_paper_ci_methods):
        upper_index_map = paper_ci_methods_tmp == paper_ci_method
        paper_upper_lengths_tmp = paper_upper_lengths[upper_index_map]
        RW_upper_lengths_tmp = RW_upper_lengths[upper_index_map]
        upper_difference = paper_upper_lengths_tmp - RW_upper_lengths_tmp
        if paper_ci_method in upper_difference_dict:
            paper_ci_method = paper_ci_method+'*'
        upper_difference_dict[paper_ci_method] = upper_difference

    # 'transpose' items to parallel key, value lists
    labels, data = [*zip(*upper_difference_dict.items())]
    labels = [label.replace('*', '') for label in labels]
    color_palette = [
        ci_method_color_dict[label.strip('*')] for label in labels]
    sns.violinplot(data, inner='quart', cut=0, palette=color_palette, scale='width')
    sns.swarmplot(data=data, color='black', alpha=0.5, size=6)
    plt.xticks(range(0, len(labels)), labels)
    plt.xticks(rotation=30, ha='right')
    plt.grid()
    plt.ylabel('Difference between original upper bound \n and upper bound estimated by \n Poisson regression with robust error variance (%)')
    plt.ylim([-10, 20])
    # if time_to_event:
    #     plt.fill_between(np.arange(-0.5, len(labels)+0.5), -10, 10, color='gray', alpha=0.2)
    plt.xlim(-0.5, len(labels)-0.5)
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
    # plt.fill_between(np.arange(-0.5, shaded_x_range+0.5), -
    #                  10, 10, color='gray', alpha=0.2)
    plt.fill_between(np.arange(-0.5, shaded_x_range+0.5), 
                     -10, 20, color='gray', alpha=0.2)
    if save_fig == True:
        plt.savefig(f'Difference_upper_RW2022.pdf')


def distance_between_VE_to_threshold(paper_vaccine_efficacy, paper_ve_methods,
                                     RW_vaccine_efficacy, save_fig=True):
    # Difference between the VE to the 50% threshold
    # Set the threshold
    threshold = 0
    palette = sns.color_palette(
        'Set1', n_colors=len(np.unique(paper_ve_methods)))

    # Remove outliers
    remove_map = RW_vaccine_efficacy != 100
    paper_vaccine_efficacy = paper_vaccine_efficacy[remove_map]
    paper_ve_methods = paper_ve_methods[remove_map]
    RW_vaccine_efficacy = RW_vaccine_efficacy[remove_map]

    # Calculate differences from the threshold
    efficacy_difference = paper_vaccine_efficacy - threshold
    recalculated_efficacy_difference = RW_vaccine_efficacy - threshold

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Efficacy Difference': np.concatenate([efficacy_difference, recalculated_efficacy_difference]),
        'Method': np.concatenate([paper_ve_methods, paper_ve_methods]),
        'Type': ['Original']*len(efficacy_difference) + ['Recalculated']*len(recalculated_efficacy_difference)
    })
    data = data.sort_values(by='Method')

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.axhline(y=50, color='black', linestyle='--')
    plt.axhline(y=30, color='black', linestyle='--')
    sns.violinplot(x='Method', y='Efficacy Difference', hue='Type',
                   data=data, inner='quart', palette=['#2ca25f', '#e5f5f9'], split=True, cut=0)

    means = data.groupby(['Method', 'Type'])['Efficacy Difference'].mean()
    
    idx = 0
    for i, mean in enumerate(means):
        if i == 0:
            plt.plot(idx-0.15, mean, '^', color='k', alpha=0.5, label='Mean')
        elif i % 2 == 0:
            plt.plot(idx-0.15, mean, '^', color='k', alpha=0.5)
        else:
            plt.plot(idx+0.15, mean, '^', color='k', alpha=0.5)
            idx += 1
        
    
    # for ind, violin in enumerate(ax.findobj(PolyCollection)):
    #     rgb = to_rgb(palette[ind // 2])
    #     if ind % 2 != 0:
    #         rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
    #     violin.set_facecolor(rgb)
    sns.swarmplot(x='Method', y='Efficacy Difference', data=data,
                  color='black', alpha=0.5, size=4)
    plt.xticks(rotation=15, ha='right')
    plt.xlabel('')
    plt.ylabel('Vaccine efficacy (%)')
    plt.fill_betweenx(y=[0, 100], x1=-0.5, x2=2.5, color='gray', alpha=0.2)
    plt.xlim(-0.5, 4.5)
    plt.ylim(20, 100)
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.grid()
    plt.legend(loc='lower right', numpoints=1)

    if save_fig:
        plt.savefig('vaccine_efficacy_threshold.pdf')
    plt.show()

    return (ax)


def distance_between_lb_to_threshold(paper_vaccine_efficacy, paper_lower_bound,
                                     paper_upper_bound, paper_ve_methods, paper_ci_methods,
                                     RW_vaccine_efficacy, RW_lower_bound, RW_upper_bound,
                                     save_fig=False):
    # Remove our 100% efficacy data and paper's 0 CI
    efficacy_remove_map = RW_vaccine_efficacy != 100
    ci_remove_map = paper_lower_bound != 0
    remove_map = efficacy_remove_map & ci_remove_map
    paper_vaccine_efficacy_tmp = paper_vaccine_efficacy[remove_map]
    paper_lower_bound_tmp = paper_lower_bound[remove_map]
    paper_upper_bound_tmp = paper_upper_bound[remove_map]
    paper_ve_methods_tmp = paper_ve_methods[remove_map]
    paper_ci_methods_tmp = paper_ci_methods[remove_map]
    RW_vaccine_efficacy_tmp = RW_vaccine_efficacy[remove_map]
    RW_lower_bound_tmp = RW_lower_bound[remove_map]
    RW_upper_bound_tmp = RW_upper_bound[remove_map]

    paper_lower_difference = np.array(paper_lower_bound_tmp, dtype=float)
    RW_lower_difference = RW_lower_bound_tmp
    ci_method_array = paper_ci_methods_tmp
    ve_method_array = paper_ve_methods_tmp
    time_to_event = np.ones(len(paper_vaccine_efficacy_tmp))*False
    for i in range(len(ve_method_array)):
        if ve_method_array[i] == 'Odds ratio' or ve_method_array[i] == 'Relative risk':
            # Forcing this to be sorted to the end
            ci_method_array[i] = 'z'+ci_method_array[i]

    data = pd.DataFrame({
        'lb_difference': np.concatenate([paper_lower_difference, RW_lower_difference]),
        'CI_method': np.concatenate([ci_method_array, ci_method_array]),
        'type': ['Original']*len(paper_lower_difference) + ['Recalculated']*len(RW_lower_difference)
    })
    data = data.sort_values(by='CI_method')


    # Plotting
    fig, ax = plt.subplots(figsize=(16, 7))
    plt.axhline(y=50, color='black', linestyle='--')
    plt.axhline(y=30, color='black', linestyle='--')
    sns.violinplot(x='CI_method', y='lb_difference', hue='type',
                   data=data, inner='quart',  palette=['#2ca25f', '#e5f5f9'], split=True, cut=0)
    sns.swarmplot(x='CI_method', y='lb_difference', data=data,
                  color='black', alpha=0.5, size=4)
    plt.fill_betweenx(y=[-10, 100], x1=-0.5, x2=8.5, color='gray', alpha=0.2)
    plt.ylim(0, 100)
    plt.xlim(-0.5, 13.5)
    # plt.xlim(-0.5, 12.5)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel('Vaccine efficacy lower bound (%)', fontsize=16)
    plt.xlabel('')
    xticks_tmp = data['CI_method'].unique()
    for i in range(len(xticks_tmp)):
        if xticks_tmp[i][0] == 'z':
            xticks_tmp[i] = xticks_tmp[i][1:]
    plt.xticks(np.arange(0, len(xticks_tmp), 1), xticks_tmp)
    # plt.xticks(np.arange(0, 13, 1), xticks_tmp)
    plt.grid()

    means = data.groupby(['CI_method', 'type'])['lb_difference'].mean()
    
    idx = 0
    for i, mean in enumerate(means):
        if i == 0:
            plt.plot(idx-0.15, mean, '^', color='k', alpha=0.5, label='Mean')
        elif i % 2 == 0:
            plt.plot(idx-0.15, mean, '^', color='k', alpha=0.5)
        else:
            plt.plot(idx+0.15, mean, '^', color='k', alpha=0.5)
            idx += 1
    plt.legend(loc='lower left', numpoints=1)
    if save_fig:
        plt.savefig('lb_threshold.pdf')
    plt.show()


############################
# visualize vaccine efficacy
############################


def plot_vaccine_efficacy(pd_data, save_figure=False):
    """
    This function plots the vaccine efficacy data for different vaccines grouping by variants type.

    Parameters
    ----------
    pd_data : pandas.DataFrame
        The DataFrame containing the vaccine efficacy data.
    save_figure : bool, optional
        Whether to save the generated figure. Default is False.

    Returns
    -------
    None

    """

    # Set color palette for the plot
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]

    # Get unique vaccine names and sort them
    vaccine_name_unique = np.sort(pd_data.vaccine.unique())

    # Initialize index and index_temp variables
    index = np.array([0])
    index_temp = 0

    # Set figure size based on the number of vaccines
    plt.figure(figsize=(10, len(pd_data)*0.35))

    # Iterate over each unique vaccine
    for i in vaccine_name_unique:
        # Get data for the current vaccine
        vaccine_data = pd_data[pd_data.vaccine == i]

        # Extract efficacy, lower bounds, and upper bounds from the data
        try:
            efficacy = vaccine_data['vaccine_efficacy']
        except:
            efficacy = vaccine_data['efficacy_in_%']
        try:
            lower_bounds = vaccine_data['lower_bound']
        except:
            lower_bounds = vaccine_data['lower']
        try:
            upper_bounds = vaccine_data['upper_bound']
        except:
            upper_bounds = vaccine_data['upper']

        # Replace 'X' in the lower and upper bounds with the efficacy value
        lower_bounds[lower_bounds.index[lower_bounds == 'X'].tolist()] = \
            efficacy[lower_bounds.index[lower_bounds == 'X'].tolist()]
        upper_bounds[upper_bounds.index[upper_bounds == 'X'].tolist()] = \
            efficacy[upper_bounds.index[upper_bounds == 'X'].tolist()]

        # Convert efficacy, lower_bounds, and upper_bounds to numpy arrays
        efficacy = efficacy.to_numpy()
        lower_bounds = lower_bounds.to_numpy()
        upper_bounds = upper_bounds.to_numpy()

        # Plot error bars for each vaccine efficacy data point
        for j in range(len(efficacy)):
            if upper_bounds[j]-efficacy[j] >= 0:
                plt.errorbar(efficacy[j], j+index[-1], xerr=np.array([[efficacy[j]-lower_bounds[j],
                                                                       upper_bounds[j]-efficacy[j]]]).T, fmt='o', color=palette[len(index)-1])
            else:
                plt.errorbar(efficacy[j], j+index[-1], xerr=np.array(
                    [[efficacy[j]-lower_bounds[j], 0]]).T, fmt='o', color=palette[len(index)-1])
            index_temp += 1
        index = np.append(index, index_temp)

    # Add vertical lines at 30% and 50% efficacy
    plt.plot([30, 30], [-1, index_temp], 'k--')
    plt.plot([50, 50], [-1, index_temp], 'k--')

    # Set labels, ticks, and limits for the plot
    plt.xlabel('Efficacy (%)', fontsize=20)
    plt.yticks(index[0:-1], vaccine_name_unique, fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlim(0, 100)
    plt.ylim([-1, index_temp])
    plt.gca().invert_yaxis()
    plt.gca().yaxis.grid()

    # Save the figure if save_figure is True
    if save_figure == True:
        plt.savefig('RW2021_Efficacy_compare.pdf',
                    format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_vaccine_efficacy_variants_group(pd_data, save_figure=False):
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    variant_type = np.sort(pd_data.variant.unique())
    for k in variant_type:  # Variant
        print(k)
        vaccine_variant_table_temp = pd_data[pd_data.variant == k]
        vaccine_name_unique = np.sort(
            vaccine_variant_table_temp.vaccine.unique())
        index = np.array([0])
        index_temp = 0
        # Forcing bars distance the same
        plt.figure(figsize=(10, len(vaccine_variant_table_temp.vaccine)*0.35))

        for i in vaccine_name_unique:  # Vaccine
            vaccine_data = vaccine_variant_table_temp[vaccine_variant_table_temp.vaccine == i]
            try:
                efficacy = vaccine_data['vaccine_efficacy']
            except:
                efficacy = vaccine_data['efficacy_in_%']
            try:
                lower_bounds = vaccine_data['lower_bound']
            except:
                lower_bounds = vaccine_data['lower']
            try:
                upper_bounds = vaccine_data['upper_bound']
            except:
                upper_bounds = vaccine_data['upper']

            # Replace 'X' in the lower and upper bound to the efficacy value
            lower_bounds[lower_bounds.index[
                (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
                = efficacy[lower_bounds.index[
                    (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
            upper_bounds[upper_bounds.index[
                (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
                = efficacy[upper_bounds.index[
                    (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

            # To numpy
            efficacy = efficacy.to_numpy()
            lower_bounds = lower_bounds.to_numpy()
            upper_bounds = upper_bounds.to_numpy()

            # Plot
            for j in range(len(efficacy)):
                # Fix the xerr negative value problem
                xerr = np.array(
                    [[max(0, efficacy[j]-lower_bounds[j]), max(0, upper_bounds[j]-efficacy[j])]]).T
                plt.errorbar(efficacy[j], j+index[-1], xerr=xerr,
                             fmt='o', color=palette[len(index)-1])
                index_temp += 1
            index = np.append(index, index_temp)

        plt.plot([30, 30], [-1, index_temp], 'k--')
        plt.plot([50, 50], [-1, index_temp], 'k--')
        plt.xlabel('Efficacy (%)', fontsize=20)
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(index[0:-1], vaccine_name_unique, fontsize=20)
        plt.ylim([-1, index_temp])
        plt.xlim([0, 100])
        plt.gca().invert_yaxis()
        plt.gca().yaxis.grid()
        if k == 'Alpha/Delta':
            k = 'Alpha_Delta'
        if k == 'Beta/Delta':
            k = 'Beta_Delta'
        if k == 'Delta/Omicron':
            k = 'Delta_Omicron'
        if save_figure == True:
            plt.savefig('RW2021_Efficacy_compare_{0}.pdf'.format(k))
        plt.show()


def plot_vaccine_efficacy_ave_group(pd_data, save_figure=False):
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    ave = pd_data.ave.unique()
    for k in ave:  # Adjusted vaccine efficacy type
        print(k)
        vaccine_ave_table_temp = pd_data[pd_data.ave == k]
        vaccine_name_unique = np.sort(
            vaccine_ave_table_temp.vaccine.unique())
        index = np.array([0])
        index_temp = 0
        # Forcing bars distance the same
        plt.figure(figsize=(10, len(vaccine_ave_table_temp.vaccine)*0.35))

        for i in vaccine_name_unique:  # Vaccine
            vaccine_data = vaccine_ave_table_temp[vaccine_ave_table_temp.vaccine == i]
            try:
                efficacy = vaccine_data['vaccine_efficacy']
            except:
                efficacy = vaccine_data['efficacy_in_%']
            try:
                lower_bounds = vaccine_data['lower_bound']
            except:
                lower_bounds = vaccine_data['lower']
            try:
                upper_bounds = vaccine_data['upper_bound']
            except:
                upper_bounds = vaccine_data['upper']

            # Replace 'X' in the lower and upper bound to the efficacy value
            lower_bounds[lower_bounds.index[
                (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
                = efficacy[lower_bounds.index[
                    (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
            upper_bounds[upper_bounds.index[
                (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
                = efficacy[upper_bounds.index[
                    (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

            # To numpy
            efficacy = efficacy.to_numpy()
            lower_bounds = lower_bounds.to_numpy()
            upper_bounds = upper_bounds.to_numpy()

            # Plot
            for j in range(len(efficacy)):
                plt.errorbar(efficacy[j], j+index[-1],
                             xerr=np.array([[max(efficacy[j]-lower_bounds[j], 0), max(upper_bounds[j]-efficacy[j], 0)]]).T, fmt='o', color=palette[len(index)-1])
                index_temp += 1
            index = np.append(index, index_temp)

        plt.plot([30, 30], [-1, index_temp], 'k--')
        plt.plot([50, 50], [-1, index_temp], 'k--')
        plt.xlabel('Efficacy (%)', fontsize=20)
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(index[0:-1], vaccine_name_unique, fontsize=20)
        plt.ylim([-1, index_temp])
        plt.xlim([0, 100])
        plt.gca().invert_yaxis()
        plt.gca().yaxis.grid()
        if k == 'Alpha/Delta':
            k = 'Alpha_Delta'
        if k == 'Beta/Delta':
            k = 'Beta_Delta'
        if k == 'Delta/Omicron':
            k = 'Delta_Omicron'
        if save_figure == True:
            plt.savefig('RW2021_Efficacy_compare_{0}.pdf'.format(k))
        plt.show()


def plot_vaccine_efficacy_ave_group_for_original(pd_data, save_figure=False):
    pd_data = pd_data[pd_data.variant == 'SARS-CoV-2']
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    ave = pd_data.ave.unique()
    for k in ave:  # Adjusted vaccine efficacy type
        print(k)
        vaccine_ave_table_temp = pd_data[pd_data.ave == k]
        vaccine_name_unique = np.sort(
            vaccine_ave_table_temp.vaccine.unique())
        index = np.array([0])
        index_temp = 0
        # Forcing bars distance the same
        plt.figure(figsize=(10, len(vaccine_ave_table_temp.vaccine)*0.35))

        for i in vaccine_name_unique:  # Vaccine
            vaccine_data = vaccine_ave_table_temp[vaccine_ave_table_temp.vaccine == i]
            try:
                efficacy = vaccine_data['vaccine_efficacy']
            except:
                efficacy = vaccine_data['efficacy_in_%']
            try:
                lower_bounds = vaccine_data['lower_bound']
            except:
                lower_bounds = vaccine_data['lower']
            try:
                upper_bounds = vaccine_data['upper_bound']
            except:
                upper_bounds = vaccine_data['upper']

            # Replace 'X' in the lower and upper bound to the efficacy value
            lower_bounds[lower_bounds.index[
                (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
                = efficacy[lower_bounds.index[
                    (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
            upper_bounds[upper_bounds.index[
                (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
                = efficacy[upper_bounds.index[
                    (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

            # To numpy
            efficacy = efficacy.to_numpy()
            lower_bounds = lower_bounds.to_numpy()
            upper_bounds = upper_bounds.to_numpy()

            # Plot
            for j in range(len(efficacy)):
                print(efficacy[j])
                print(lower_bounds[j], upper_bounds[j])
                if lower_bounds[j] == 0 and upper_bounds[j] == 0:
                    plt.errorbar(efficacy[j], j+index[-1],
                                 xerr=np.array([[0, 0]]).T, fmt='o', color=palette[len(index)-1])
                else:
                    plt.errorbar(efficacy[j], j+index[-1],
                                 xerr=np.array([[max(0, efficacy[j]-lower_bounds[j]),
                                                max(0, upper_bounds[j]-efficacy[j])]]).T, fmt='o', color=palette[len(index)-1])
                index_temp += 1
            index = np.append(index, index_temp)

        plt.plot([30, 30], [-1, index_temp], 'k--')
        plt.plot([50, 50], [-1, index_temp], 'k--')
        plt.xlabel('Efficacy (%)', fontsize=20)
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(index[0:-1], vaccine_name_unique, fontsize=20)
        plt.ylim([-1, index_temp])
        plt.xlim([0, 100])
        plt.gca().invert_yaxis()
        plt.gca().yaxis.grid()
        if k == 'Alpha/Delta':
            k = 'Alpha_Delta'
        if k == 'Beta/Delta':
            k = 'Beta_Delta'
        if k == 'Delta/Omicron':
            k = 'Delta_Omicron'
        if k == 'Documented infection':
            k = 'Documented_infection'
        if save_figure == True:
            plt.savefig('RW2021_Efficacy_compare_SARS-CoV-2_{0}.pdf'.format(k))
        plt.show()


def plot_vaccine_efficacy_ave_group_for_variants(pd_data, save_figure=False):
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    pd_data = pd_data[pd_data.variant != 'SARS-CoV-2']  # Remove original virus
    ave = pd_data.ave.unique()
    for k in ave:  # Adjusted vaccine efficacy type
        print(k)
        vaccine_ave_table_temp = pd_data[pd_data.ave == k]
        vaccine_name_unique = np.sort(
            vaccine_ave_table_temp.vaccine.unique())
        index = np.array([0])
        index_temp = 0
        variant_names = np.array([])
        # Forcing bars distance the same
        fig = plt.figure(
            figsize=(10, len(vaccine_ave_table_temp.vaccine)*0.35))
        ax = fig.add_subplot(111)
        for i in vaccine_name_unique:  # Vaccine
            vaccine_data = vaccine_ave_table_temp[vaccine_ave_table_temp.vaccine == i]
            try:
                efficacy = vaccine_data['vaccine_efficacy']
            except:
                efficacy = vaccine_data['efficacy_in_%']
            try:
                lower_bounds = vaccine_data['lower_bound']
            except:
                lower_bounds = vaccine_data['lower']
            try:
                upper_bounds = vaccine_data['upper_bound']
            except:
                upper_bounds = vaccine_data['upper']

            # Replace 'X' in the lower and upper bound to the efficacy value
            lower_bounds[lower_bounds.index[
                (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
                = efficacy[lower_bounds.index[
                    (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
            upper_bounds[upper_bounds.index[
                (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
                = efficacy[upper_bounds.index[
                    (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

            # To numpy
            efficacy = efficacy.to_numpy()
            lower_bounds = lower_bounds.to_numpy()
            upper_bounds = upper_bounds.to_numpy()

            # Extract variant name
            variant_names_temp = vaccine_data.variant.to_numpy()
            variant_names = np.append(variant_names, variant_names_temp)

            # Plot
            for j in range(len(efficacy)):
                print(efficacy[j])
                print(lower_bounds[j], upper_bounds[j])
                if lower_bounds[j] == 0 and upper_bounds[j] == 0:
                    plt.errorbar(efficacy[j], j+index[-1],
                                 xerr=np.array([[0, 0]]).T, fmt='o', color=palette[len(index)-1])
                else:
                    plt.errorbar(efficacy[j], j+index[-1],
                                 xerr=np.array([[max(0, efficacy[j]-lower_bounds[j]),
                                                max(0, upper_bounds[j]-efficacy[j])]]).T, fmt='o', color=palette[len(index)-1])
                index_temp += 1
            index = np.append(index, index_temp)

        ax.plot([30, 30], [-1, index_temp], 'k--')
        ax.plot([50, 50], [-1, index_temp], 'k--')
        ax.set_xlabel('Efficacy (%)', fontsize=20)
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_yticks(index[0:-1])
        ax.set_yticklabels(vaccine_name_unique, fontsize=20)
        ax.set_ylim([-1, index_temp])
        ax.set_xlim([0, 100])
        plt.gca().invert_yaxis()
        plt.gca().yaxis.grid()
        ax2 = ax.twinx()
        ax2.set_yticks(np.arange(index[0], index[-1]))
        ax2.set_yticklabels(variant_names)
        ax2.set_ylim([-1, index_temp])
        plt.gca().invert_yaxis()

        if k == 'Alpha/Delta':
            k = 'Alpha_Delta'
        if k == 'Beta/Delta':
            k = 'Beta_Delta'
        if k == 'Delta/Omicron':
            k = 'Delta_Omicron'
        if save_figure == True:
            plt.savefig('RW2021_Efficacy_compare_variants_{0}.pdf'.format(k))
        plt.show()


def plot_vaccine_efficacy_variants_and_ave_group(pd_data, save_figure=False):
    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.vaccine.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    ave = pd_data.ave.unique()
    for m in ave:
        print(m)
        vaccine_table_temp = pd_data[pd_data.ave == m]
        variant_type = np.sort(vaccine_table_temp.variant.unique())
        for k in variant_type:
            print(k)
            vaccine_variant_table_temp = vaccine_table_temp[vaccine_table_temp.variant == k]
            vaccine_name_unique = np.sort(
                vaccine_variant_table_temp.vaccine.unique())
            index = np.array([0])
            index_temp = 0
            # Forcing bars distance the same
            plt.figure(
                figsize=(10, len(vaccine_variant_table_temp.vaccine)*0.35))
            for i in vaccine_name_unique:  # Vaccine
                vaccine_data = vaccine_variant_table_temp[vaccine_variant_table_temp.vaccine == i]
                try:
                    efficacy = vaccine_data['vaccine_efficacy']
                except:
                    efficacy = vaccine_data['efficacy_in_%']
                try:
                    lower_bounds = vaccine_data['lower_bound']
                except:
                    lower_bounds = vaccine_data['lower']
                try:
                    upper_bounds = vaccine_data['upper_bound']
                except:
                    upper_bounds = vaccine_data['upper']

                # Replace 'X' in the lower and upper bound to the efficacy value
                lower_bounds[lower_bounds.index[
                    (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
                    = efficacy[lower_bounds.index[
                        (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
                upper_bounds[upper_bounds.index[
                    (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
                    = efficacy[upper_bounds.index[
                        (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

                # To numpy
                efficacy = efficacy.to_numpy()
                lower_bounds = lower_bounds.to_numpy()
                upper_bounds = upper_bounds.to_numpy()

                # Plot
                for j in range(len(efficacy)):
                    xerr = np.array(
                        [[max(0, efficacy[j]-lower_bounds[j]), max(0, upper_bounds[j]-efficacy[j])]]).T
                    plt.errorbar(
                        efficacy[j], j+index[-1], xerr=xerr, fmt='o', color=palette[len(index)-1])
                    index_temp += 1
                index = np.append(index, index_temp)
            plt.plot([30, 30], [-1, index_temp], 'k--')
            plt.plot([50, 50], [-1, index_temp], 'k--')
            plt.xlabel('Efficacy (%)', fontsize=20)
            plt.xticks(np.arange(0, 110, 10))
            plt.yticks(index[0:-1], vaccine_name_unique, fontsize=20)
            plt.ylim([-1, index_temp])
            plt.xlim([0, 100])
            plt.gca().invert_yaxis()
            plt.gca().yaxis.grid()
            if k == 'Alpha/Delta':
                k = 'Alpha_Delta'
            if k == 'Beta/Delta':
                k = 'Beta_Delta'
            if k == 'Delta/Omicron':
                k = 'Delta_Omicron'
            if save_figure == True:
                plt.savefig('RW2021_Efficacy_compare_%s_%s.pdf' % (k, m))
            plt.show()


def plot_average_vaccine_efficacy(pd_data, save_figure=False):
    try:
        pd_data = pd_data[pd_data.vaccine == 'All vaccine']
    except:
        print('Missing All vaccine')
    # Drop mixed variant
    pd_data = pd_data[pd_data.variant != 'Mixed variants']

    palette = sns.color_palette(
        'Set1', n_colors=len(pd_data.ave.unique())+2)
    palette = palette[0:5] + palette[6:14] + palette[15::]
    index = np.arange(len(pd_data))
    # endpoint_unique = np.sort(pd_data.ave.unique())
    endpoint_unique = np.array(
        ['Asymptomatic', 'Symptomatic', 'Moderate', 'Severe', 'Critical', 'Death'])

    # Forcing bars distance the same
    plt.figure(figsize=(10, len(pd_data.ave)*0.35))
    for i, endpoint in enumerate(endpoint_unique):
        vaccine_data = pd_data[pd_data.ave == endpoint]
        try:
            efficacy = vaccine_data['vaccine_efficacy']
        except:
            efficacy = vaccine_data['efficacy_in_%']
        try:
            lower_bounds = vaccine_data['lower_bound']
        except:
            lower_bounds = vaccine_data['lower']
        try:
            upper_bounds = vaccine_data['upper_bound']
        except:
            upper_bounds = vaccine_data['upper']

        # Replace 'X' in the lower and upper bound to the efficacy value
        lower_bounds[lower_bounds.index[
            (lower_bounds == 'X') | (lower_bounds == '-')].tolist()] \
            = efficacy[lower_bounds.index[
                (lower_bounds == 'X') | (lower_bounds == '-')].tolist()]
        upper_bounds[upper_bounds.index[
            (upper_bounds == 'X') | (upper_bounds == '-')].tolist()] \
            = efficacy[upper_bounds.index[
                (upper_bounds == 'X') | (upper_bounds == '-')].tolist()]

        # To numpy
        efficacy = efficacy.to_numpy()
        lower_bounds = lower_bounds.to_numpy()
        upper_bounds = upper_bounds.to_numpy()

        # Plot
        plt.errorbar(efficacy, index[i], xerr=np.array(
            [efficacy-lower_bounds, upper_bounds-efficacy]), fmt='o', color=palette[index[i]])

    plt.plot([30, 30], [-1, index[-1]], 'k--')
    plt.plot([50, 50], [-1, index[-1]], 'k--')
    plt.xlabel('Efficacy (%)', fontsize=20)
    plt.xticks(np.arange(0, 110, 10))
    endpoint_y = ['Asymptomatic (3)', 'Symptomatic (16)',
                  'Moderate (4)', 'Severe (14)', 'Critical (4)', 'Death (4)']
    plt.yticks(index, endpoint_y, fontsize=20)
    plt.ylim([-1, index[-1]+0.1])
    plt.xlim([0, 100])
    plt.gca().invert_yaxis()
    plt.gca().yaxis.grid()

    if save_figure == True:
        plt.savefig('RW2021_Avg_efficacy_compare.pdf')
    plt.show()
