from logger import Logger

def format_to_str(alist):
    s = "["
    if isinstance(alist,list):
        for i in range(len(alist)):
            item = alist[i]
            if isinstance(item,list):
                s = s + format_to_str(item)
            else:
                s = s + str(item)
            if i < len(alist)-1:
                    s += ","
                
        
    s = s + "]"
    return s


if __name__ == "__main__":
    logger = Logger()
    logger.debug(format_to_str(["1",2,"a","d"]))
    logger.debug(format_to_str(["1",2,['a','b',3]]))
    logger.debug(format_to_str(["1","2",['a',['d','e'],'c']]))