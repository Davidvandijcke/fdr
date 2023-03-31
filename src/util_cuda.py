def getParam(param, var, argc, argv):
    c_param = param.c_str()
    for i in range(argc-1, 1, -1):
        if argv[i][0] != '-': continue
        if strcmp(argv[i]+1, c_param) == 0:
            if not (i+1 < argc) or argv[i+1][0] == '-':
                var = True
                return True
            ss = stringstream()
            ss << argv[i+1]
            ss >> var
            return bool(ss)
    return False