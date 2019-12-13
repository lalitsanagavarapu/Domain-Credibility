#Author : Avi Solanki

# Sorted directly w.r.t. similarity

    # w1 = avg_feature_vector(keyword, model=model, num_features=150, index2word_set=index2word_set)
    # w11 = avg_feature_vector(word_with_spaces, model=model, num_features=150, index2word_set=index2word_set)
    # raw_children = []
    # sim_children = []
    # children = []
    # simi_sort = []

    # for term in ms:
    #     term_name = term.name.lower()
    #     term_name = ""
    #     if term_name == "":
    #         term_repr = term.__repr__()
    #         for ind,c in enumerate(term_repr):
    #             if term_repr[ind] == ':':
    #                 break
    #             if ind-1 and c.isupper() and (term_repr[ind+1].islower() or term_repr[ind+1]==':'):
    #                 term_name += " "
    #             if c != '<':
    #                 term_name += c
    #     term_name = term_name.lower()
    #     if term_name == "":
    # #         print (term.__repr__())
    #         continue
    #     w2 = avg_feature_vector(term_name, model=model, num_features=150, index2word_set=index2word_set)
    #     neg_sim1 = spatial.distance.cosine(w1, w2)
    #     if math.isnan(neg_sim1):
    #         neg_sim1 = 1
    #     neg_sim11 = spatial.distance.cosine(w11, w2)
    #     if math.isnan(neg_sim11):
    #         neg_sim11 = 1
    # #     print (word_with_spaces,term_name,term.name,1-min(neg_sim1,neg_sim11))
    # #     if term_name == "firewall a":
    # #         print (term.__repr__(),min(neg_sim1,neg_sim11))
    #     sim_children.append(min(neg_sim1,neg_sim11))
    #     raw_children.append(term_name)
        
    # children = [raw_children for _,raw_children in sorted(zip(sim_children,raw_children))]
    # sim_children.sort()
    # # print (children)
    # for i,child in enumerate(children):
    #     print (child,1-sim_children[i])