# This function creates the plots comparing the input and output fom the Halo-Property Autoencoder
# denorm_input is the de-normalized input features from test set
# denorm_output is the de-normalized output features reproduced by autoencoder
# fname is the name of the file containing the saved plot

def make_plots(denorm_input, denorm_output, fname):
    property_name = ["m_vir", "v_max", "v_rms", "Halo Radius", "Scale Radius", 
                 "velocity", "Angular momentum", "Spin", "b_to_a",
                 "c_to_a", "T/|U|"]
    n_plots = len(property_name)
    rows    = n_plots
    columns = 2
    fig = plt.figure(figsize=(17, 75))
    
    # ax enables access to manipulate each of subplots
    ax = []
    
    for i in range(n_plots):
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(property_name[i])  # set title
        
        # y=x plot
        min = np.min([np.min(denorm_input[:,i]), np.min(denorm_output[:,i])])
        max = np.max([np.max(denorm_input[:,i]), np.max(denorm_output[:,i])])
        x = np.linspace(min, max, 1000)        
        
        # Plot the input and predicted values
        plt.scatter(denorm_input[:,i], denorm_output[:,i])
        plt.plot(x, x, '-r')
        plt.title(property_name[i], fontsize = 15)
        plt.ylabel("Predicted Value", fontsize = 15)
        plt.xlabel("Input Value", fontsize = 15)
        
        # Compute r_squared value
        r_squared_value = r_squared(denorm_input[:,i], denorm_output[:,i])
        textstr = str(r_squared_value)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        x_max = np.max(denorm_input[:,i])
        x_min = np.min(denorm_input[:,i])
        plt.text(np.median([x_max,x_min]), min,textstr, fontsize=14, bbox=props, horizontalalignment="center") 
        
        if i == 0 or i == 6:
            plt.yscale('log')
            plt.xscale('log')
    plt.savefig(fname)
    plt.show()
