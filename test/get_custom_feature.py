def _load_next(data_config, filelist, load_range, options):
    load_branches = data_config.train_load_branches if options['training'] else data_config.test_load_branches
    table = _read_files(filelist, load_branches, load_range, treename=data_config.treename,
                        branch_magic=data_config.branch_magic, file_magic=data_config.file_magic)
    table = get_custom_feature(table)  
    table, indices = _preprocess(table, data_config, options)
    return table, indices


def get_custom_feature(table):
    table['part_dEta_two_particles '] = get_two_fea(table['part_eta'])
    table['part_dPhi_two_particles'] = get_dPhi(table['part_phi'])
    table['jet_dEta_two_jets'] = get_two_fea(table['jet_eta'])
    table['jet_dPhi_two_jets'] = get_dPhi(table['jet_phi'])
    table['part_dR_two_particles'] = get_squares_sum(table['part_dEta_two_particles'], table['part_dPhi_two_particles'])
    table['jet_dR_two_jets'] = get_squares_sum(table['jet_dEta_two_jets'], table['jet_dPhi_two_jets'])
    table['part_logptrel_particle_jet'] = get_jet_fea(
        table['part_pt'], table['jet_pt'], table['part_slimjetid'], table['part_fatjetid'])
    table['part_logerel_particle_jet'] = get_jet_fea(
        table['part_energy'], table['jet_energy'], table['particle_slimjetid'], table['particle_fatjetid'])
    table['part_dEta_particle_jet'] = get_two_fea(get_jet_fea(
        table['part_eta'], table['jet_eta'], table['particle_slimjetid'], table['particle_fatjetid']))
    table['part_dPhi_particle_jet'] = get_dPhi(get_jet_fea(
        table['part_phi'], table['jet_phi'], table['particle_slimjetid'], table['particle_fatjetid']))
    table['part_dR_particle_jet'] = get_squares_sum(table['part_dEta_particle_jet'], table['part_dPhi_particle_jet'])

    table['part_logptrel_particle_event'] = get_event_fea(table['part_pt'], table['event_pt'])
    table['part_logerel_particle_event'] = get_event_fea(table['part_energy'], table['event_energy'])
    table['jet_logptrel_jet_event'] = get_event_fea(table['jet_pt'], table['event_pt'])
    table['jet_logerel_jet_event'] = get_event_fea(table['jet_energy'], table['event_energy'])
    table['part_dEta_particle_event'] = get_two_fea(get_event_fea(table['part_eta'], table['event_eta']))
    table['part_dPhi_particle_event'] = get_dPhi(get_event_fea(table['part_phi'], table['event_phi']))
    table['jet_dEta_jet_event'] = get_two_fea(get_event_fea(table['jet_eta'], table['event_eta']))
    table['jet_dPhi_jet_event'] = get_dPhi(get_event_fea(table['jet_phi'], table['event_phi']))

    table['part_dR_particle_event'] = get_squares_sum(
        table['part_dEta_particle_event'], table['part_dPhi_particle_event'])
    table['jet_dR_jet_event'] = get_squares_sum(
        table['jet_dEta_jet_event'], table['jet_dPhi_jet_event'])
    return table


def get_two_fea(input_features):
    length = len(input_features) 
    result = []
    for i in range(0, length):
        curr_sample_len = len(input_features[i]) 
        curr_res = [] 
        for j in range(curr_sample_len - 1):
            for k in range(j + 1, curr_sample_len):
                new_value = input_features[i][j] - input_features[i][k] 
                curr_res.append(new_value)
        result.append(curr_res)
    return result


def get_squares_sum(features1, features2):
    length = len(features1)  
    result = []
    for i in range(0, length):
        curr_sample_len = len(features1[i])  
        curr_res = []  
        for j in range(curr_sample_len):
            curr_res.append(math.sqrt(math.pow(features1[i][j], 2) + math.pow(features2[i][j], 2)))
        result.append(curr_res)
    return result


def get_jet_fea(part_features, jet_features, slimjet_ids, fatjet_ids):
    length = len(part_features)  
    result = []
    for i in range(0, length):
        curr_result = []  
        id2indexs = get_pair_res(slimjet_ids[i], fatjet_ids[i])
        for curr_id, index_list in id2indexs:
            for curr_index in index_list:
                curr_result.append(
                    math.log(part_features[i][curr_index] / jet_features[i][curr_id]))
        result.append(curr_result)
    return result


def get_event_fea(data_features, event_features):
    length = len(data_features)  
    result = []
    for i in range(0, length):
        curr_sample_len = len(data_features[i])  
        curr_result = []  
        for j in range(curr_sample_len):
            curr_result.append(
                math.log(data_features[i][j] / event_features[i]))   
        result.append(curr_result)
    return result


def get_pair_res(slimjet_id, fatjet_id):
    res = {}
    for index, curr_id in enumerate(slimjet_id):
        if curr_id == 0:
            continue
        if curr_id in res:
            res[curr_id].append(index)
        else:
            res[curr_id] = [index]
    for index, curr_id in enumerate(fatjet_id):
        if curr_id == 0:
            continue
        if curr_id in res:
            res[curr_id].append(index)
        else:
            res[curr_id] = [index]
    sorted_res = sorted(res.items(), key=lambda x: x[0])
    return sorted_res
    
    
def get_dPhi(input_features):
    length = len(input_features) 
    result = []
    for i in range(0, length):
        curr_sample_len = len(input_features[i]) 
        curr_res = [] 
        for j in range(curr_sample_len - 1):
            for k in range(j + 1, curr_sample_len):
                new_value = input_features[i][j] - input_features[i][k] 
                while new_value > math.pi:
                    new_value -= 2 * math.pi
                while new_value < -math.pi:
                    new_value += 2 * math.pi
                curr_res.append(new_value)
        result.append(curr_res)
    return result
