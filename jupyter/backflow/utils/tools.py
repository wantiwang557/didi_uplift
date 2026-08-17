import datetime

def generate_dt_list(start_dt, degree_days, step=1, format="%Y-%m-%d", flag=1):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    if flag == 1:
        return [strftime(strptime(start_dt, format) + datetime.timedelta(i), format) for i in range(0, degree_days, step)]
    return [strftime(strptime(start_dt, format) - datetime.timedelta(i), format) for i in range(0, degree_days, step)][::-1]

def date2string(base_dt, days, flag=-1):
    base_dt   = datetime.datetime.strptime(base_dt, "%Y-%m-%d")
    target_dt = base_dt - datetime.timedelta(days=days) if flag == -1 else base_dt + datetime.timedelta(days=days)
    target_dt = target_dt.strftime("%Y-%m-%d")

    return target_dt


def smart_generate_dt_list(start_dt, degree_days, step=1, format="%Y-%m-%d", flag=1, 
                          ignore_dts=[]):
    
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    
    gen_dts = []
    curr_dt = start_dt
    
    while len(gen_dts)<degree_days:
        if curr_dt not in ignore_dts:
            gen_dts.append(curr_dt)
        
        __dt = strftime(strptime(curr_dt, format) - datetime.timedelta(1), format)
        if flag == 1:
            __dt = strftime(strptime(curr_dt, format) + datetime.timedelta(1), format)
            
        curr_dt = __dt
    
        
    gen_dts = sorted(gen_dts)
    if gen_dts[-1] != start_dt:
        gen_dts.append(start_dt)
    return sorted(gen_dts)[-degree_days:]