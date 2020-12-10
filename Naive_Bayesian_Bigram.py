# <!-- // Code Author:-
# // Name: Shivam Gupta
# // Net ID: SXG190040
# // Natural Language Processing  (CS 6320.501) Assignment 3 (Question 1 - Naive Bayesian Bigram Model) -->

#Function for Reading the file
def READ_FILE(path):
    FIL = open(path, "r")
    FIL = FIL.read()
    return FIL

#Function for extracting each word and tag
def PRE_Processing(Sent):
    word_tag_pair = Sent.split()
    for i in range(0, len(word_tag_pair)):
        word_tag_pair[i] = word_tag_pair[i].split('_')
    return word_tag_pair


def BiGram_Model(input,LINES):
    #calculate all the tag counts
    TAGs_DICT=dict()
    for line in LINES:
        line=PRE_Processing(line)
        for WORD_TAG in line:
            if WORD_TAG[1] not in TAGs_DICT:
                # Set it to one
                TAGs_DICT[WORD_TAG[1]] =1
            else:
                # Adding unigram count
                TAGs_DICT[WORD_TAG[1]] += 1
    #print(TAGs_DICT)

    #calculate all the words given TAGS counts
    WORDS_given_TAGS_DICT={}
    for line in LINES:
        line = PRE_Processing(line)
        for WORD_TAG in line:
            #print(words_and_tags[0],words_and_tags[1])
            Wo_Tag=(WORD_TAG[0],WORD_TAG[1])
            if Wo_Tag not in WORDS_given_TAGS_DICT:
                WORDS_given_TAGS_DICT[Wo_Tag]=1
            else:
                WORDS_given_TAGS_DICT[Wo_Tag]+=1
    #print(WORDS_given_TAGS_DICT)

    # Calculating the probabilty of word given TAGS = P(W|T)=p(W&T)/p(T)
    PROBS_WORD_given_TAG_DICT={}
    for Wo_Tag in WORDS_given_TAGS_DICT:
        PROBS_WORD_given_TAG_DICT[Wo_Tag]=WORDS_given_TAGS_DICT.get(Wo_Tag)/TAGs_DICT.get(Wo_Tag[1])


    BIGRAMs_TAGs_DICT=dict()
    # Creating the Bigram model of the TAGS(Consecutive Tags one after the other):
    for line in LINES:
        line=PRE_Processing(line)


        for i in range(len(line)-1):
            BIGRAMs_TAG=(line[i+1][1],line[i][1])
            if BIGRAMs_TAG not in BIGRAMs_TAGs_DICT:
                BIGRAMs_TAGs_DICT[BIGRAMs_TAG]=1
            else:
                BIGRAMs_TAGs_DICT[BIGRAMs_TAG]+=1
    #print (BIGRAMs_TAGs_DICT)

    PROBS_TAG_given_TAG_DICT={}

    for BIGRAMs_TAG in BIGRAMs_TAGs_DICT:
        #print(BIGRAMs_TAG)
        #print(BIGRAMs_TAG[1])
        PROBS_TAG_given_TAG_DICT[BIGRAMs_TAG]=BIGRAMs_TAGs_DICT.get(BIGRAMs_TAG)/TAGs_DICT.get(BIGRAMs_TAG[1])



    return WORDS_given_TAGS_DICT,PROBS_WORD_given_TAG_DICT,PROBS_TAG_given_TAG_DICT


def Read_Lines(Training_File):
    FIL = open(Training_File, "r")
    FIL = FIL.readlines()
    New_LINES=[]
    for line in FIL:
        New_L="<S>_<S> "+line+" </S>_</S>"
        New_LINES.append(New_L)
    return New_LINES


def Naive_Bayesian_Model(TEST_Sentence,WORDS_given_TAGS_DICT,PROBS_WORD_given_TAG_DICT, PROBS_TAG_given_TAG_DICT):
    SENT = TEST_Sentence.split()
    for i in range(0, len(SENT)):
        SENT[i] = SENT[i].split('_')

    # Finding the Possible TAGS of the SENT
    KEYS = list()
    for i in WORDS_given_TAGS_DICT.keys():
        KEYS.append(i)

    #dictionary of key and its possible TAGS
    Possible_TAGS= dict()
    s=1
    for word in SENT:
        TAGS_Array=[]
        for key in KEYS:
            if(word[0]==key[0]):
                TAGS_Array.append(key[1])

        if(len(TAGS_Array)>0):
            Possible_TAGS[word[0]+" "+str(s)]=TAGS_Array
            s+=1
    print("---------------------------------------------------")
    print("Words with the Possible TAGS are as follows")
    print(Possible_TAGS)
    print("---------------------------------------------------")


    TAG_TAG_SEQ=[]
    WORD_TAG_PAIR={}
    i=1
    for key in Possible_TAGS:
        keys=key.split()
        # If there is only 1 Possibility of the Tag
        if(len(Possible_TAGS.get(key))==1):
            pair=(key,Possible_TAGS.get(key)[0])
            WORD_TAG_PAIR[i]=pair
            TAG_TAG_SEQ.append(Possible_TAGS.get(key)[0])

        # If there is more than 1 Possibility of the Tag
        else:
            MAX_W_T_Prob_Dict= dict()
            MAX_T_T_Prob_Dict= dict()
            # PREVIOUS_W_T is the word with Tag
            PREVIOUS_W_T=WORD_TAG_PAIR.get(i-1)
            #print(PREVIOUS_W_T[0].split()[0])
            PREV_word_possible_TAGS=Possible_TAGS.get(PREVIOUS_W_T[0].split()[0]+" "+str(i-1))
            for TAGS in Possible_TAGS.get(key):
                pair=(keys[0],TAGS)
                New_Pair=(key,TAGS)
                MAX_W_T_Prob_Dict[New_Pair]=PROBS_WORD_given_TAG_DICT.get(pair)

            for TAGS in Possible_TAGS.get(key):
                for PREV_TAG in PREV_word_possible_TAGS:
                    pair=(TAGS,PREV_TAG)
                    MAX_T_T_Prob_Dict[pair]=PROBS_TAG_given_TAG_DICT.get(pair)

            MAX_Tag_prob=0
            MAX_TAG=()
            MAX_Wo=()
            for tag_tag in MAX_T_T_Prob_Dict:
                for WORD_TAG in MAX_W_T_Prob_Dict:
                    if((tag_tag[0]==WORD_TAG[1]) and (MAX_W_T_Prob_Dict.get(WORD_TAG) is not None) and (MAX_T_T_Prob_Dict.get(tag_tag)is not None)):
                        #print(WORD_TAG)
                        Naive_Product=MAX_W_T_Prob_Dict.get(WORD_TAG)*MAX_T_T_Prob_Dict.get(tag_tag)
                        if(Naive_Product>MAX_Tag_prob):
                            MAX_Tag_prob=Naive_Product
                            MAX_TAG=tag_tag
                            MAX_Wo=WORD_TAG


            # print(TAG_TAG_SEQ)
            # print(WORD_TAG_PAIR)
            # Appending the Predicted tag sequence
            TAG_TAG_SEQ.append(MAX_TAG[0])
            WORD_TAG_PAIR[i] = (MAX_Wo)
            # print(MAX_T_T_Prob_Dict)
            #print(MAX_W_T_Prob_Dict)
            # print(MAX_TAG)
            #print(MAX_Wo)
        i = i + 1

    WORD_TAG_PAIR.pop(1)
    WORD_TAG_PAIR.pop(len(WORD_TAG_PAIR)+1)



    Final_W_T_Prob_DICT= dict()
    # Creating the word tag probabilities
    for key in WORD_TAG_PAIR.keys():
        W_T_pair=WORD_TAG_PAIR.get(key)
        W_T_pair=(W_T_pair[0].split()[0],W_T_pair[1])
        Final_W_T_Prob_DICT[WORD_TAG_PAIR.get(key)]=PROBS_WORD_given_TAG_DICT.get(W_T_pair)
    # Creating the Bigrams
    T_T_BiGr_DICT= dict()
    for i in range(len(TAG_TAG_SEQ)-1):
        tag_pair=(TAG_TAG_SEQ[i+1],TAG_TAG_SEQ[i])
        # tag_pair=(TAG_TAG_SEQ[i],TAG_TAG_SEQ[i+1])
        T_T_BiGr_DICT[tag_pair]=PROBS_TAG_given_TAG_DICT.get(tag_pair)


    print("\n The probabilities of the given tag sequence are and the Best tag sequence is: \n")
    # print (T_T_BiGr_DICT)
    print("------------------------------------------------------------------------------")
    
    FINAL_W_T_Prob_DICT = {}
    for k in Final_W_T_Prob_DICT.keys():
        new_key = (k[0].split()[0],k[1])
        FINAL_W_T_Prob_DICT[new_key] = Final_W_T_Prob_DICT[k]

    print(" \n The word with Best POS tag and Their probabilities are \n")
    for item in FINAL_W_T_Prob_DICT.items():
        print (item)
    T_List =  []
    for wt in list(Final_W_T_Prob_DICT.keys()):
        T_List.append(wt[1])
    
    T_List.insert(0, '<S>')
    T_List.append('</S>')
    print("\n ======= The words and Best POS TAGS are ========= \n")
    print(TEST_Sentence)
    print(T_List)


    Output_PROB=1


    for key in Final_W_T_Prob_DICT:
        Output_PROB *= Final_W_T_Prob_DICT.get(key)
        # print(Output_PROB)
    for key in T_T_BiGr_DICT:
        # print("T_T_BiGr_DICT.get(key)",key,  T_T_BiGr_DICT.get(key))
        if T_T_BiGr_DICT[key] is None:
            continue
        else:
            Output_PROB = Output_PROB*T_T_BiGr_DICT.get(key)

    print("---------------------------------------------------")
    print(" \n The Final probabilty after Naïve Bayesian Classification POS tagging is: \n")
    print(Output_PROB)
    print("----------------*********--------------------------")

# Main Function
if __name__ == "__main__":
    Training_File = 'NLP6320_POSTaggedTrainingSet.txt'
    FIL = READ_FILE(Training_File)
    LINES  = Read_Lines(Training_File)
    WORD_TAG = PRE_Processing(FIL)
    # TRAINING The Bigram Model
    WORDS_given_TAGS_DICT,PROBS_WORD_given_TAG_DICT,PROBS_TAG_given_TAG_DICT = BiGram_Model(WORD_TAG,LINES)
    
    print(" Naïve Bayesian POS Training Completed \n ")
    TEST_Sentence=input("Please Enter a sentance for TESTING of POS Tagging: \n")
    # TEST_Sentence="Brainpower , not physical plant , is now a firm 's chief asset ."
    TEST_Sentence="<S> "+TEST_Sentence+" </S>"
    print("\n")
    print("=======The Test Sentence is=====", TEST_Sentence)
    # TESTING The Bigram Model
    print("\n Naïve Bayesian Classification (Bigram) based POS Tagging \n ")
    Naive_Bayesian_Model(TEST_Sentence,WORDS_given_TAGS_DICT,PROBS_WORD_given_TAG_DICT,PROBS_TAG_given_TAG_DICT)