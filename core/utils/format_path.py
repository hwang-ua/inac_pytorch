import os


def fill_run_number(dic, run_num, param_num, data_root):
    if 'load_params' in dic.keys() and dic['load_params']:
        if type(dic['path']) == dict:
            for key in dic['path']:
                dic['path'][key] = dic['path'][key].format(run_num, param_num)
                if len(dic['path'][key].split("{}")) == 2:
                    print("Filling in run number only")
                path = os.path.join(data_root, dic['path'][key])
                if not os.path.isfile(path):
                    print("Run {} doesn't exist. {}".format(run_num, path))
                    exit(1)
        else:
            if len(dic['path'].split("{}")) == 2:
                print("Filling in run number only")
            dic['path'] = dic['path'].format(run_num, param_num)

            path = os.path.join(data_root, dic['path'])
            if not os.path.isfile(path):
                print("Run {} doesn't exist. {}".format(run_num, path))
                exit(1)
    else:
        print("Not Loading Parameter for {}".format(dic))
    return dic
    