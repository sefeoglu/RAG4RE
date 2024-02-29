

def get_zero_shot_template_tacred(sentence, relation, head, tail):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """

    template_zero_shot = """Question : What is the relation type between head and tail entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ Head entity: """ + head + """. \n""" +\
                        """ Tail entity: """ + tail + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot

def get_zero_shot_template_tacred_rag(sentence, relation, head, tail, context):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """

    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Question : What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
                        """ Example Sentence: """+ str(context)+ """\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ tail: """ + head + """. \n""" +\
                        """ head: """ + tail + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot

    

def semeval_prompt_template_rag(sentence, relation, head, tail, head_name, tail_name, context):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Question : What is the relation type between """+head+""" and """+tail+""" entities according to given relationships below in the following sentence, considering example sentence and its relationship?\n""" +\
                        """ Example Sentence:"""+str(context)+ """\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot

def semeval_prompt_template(sentence, relation, head, tail, head_name, tail_name):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """
    template_zero_shot = """Question : What is the relation type between """+head+""" and """+tail+""" entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot