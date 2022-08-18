import fungi_classification

if __name__ == '__main__':
    # Your team and team password
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "SwimmingApe"
    team_pw = "fungi18"

    # where is the full set of images placed
    #image_dir = "C:/data/Danish Fungi/DF20M/"
    image_dir = "/scratch/hilss/DF20M/"

    # where should log files, temporary files and trained models be placed
    #network_dir = "C:/data/Danish Fungi/FungiNetwork/"
    network_dir = "/scratch/kmze/FungiNetworkKilian"

    
    #Create the Pool CSV and predict with current model
    fungi_classification.create_pool_csv(team, team_pw, image_dir, network_dir)
    fungi_classification.variation_ratio_label(team, team_pw,  network_dir)
    