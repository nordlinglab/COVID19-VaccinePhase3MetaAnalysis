import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt





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
    palette = sb.color_palette('Set1', n_colors=len(pd_data.vaccine.unique())+2)
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
        plt.savefig('RW2021_Efficacy_compare.pdf', format='pdf', bbox_inches='tight')
    
    # Display the plot
    plt.show()


def plot_vaccine_efficacy_variants_group(pd_data, save_figure=False):
    palette = sb.color_palette(
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
    palette = sb.color_palette(
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
    palette = sb.color_palette(
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
    palette = sb.color_palette(
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
    palette = sb.color_palette(
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

    palette = sb.color_palette(
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