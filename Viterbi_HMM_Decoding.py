# <!-- // Code Author:-
# // Name: Shivam Gupta
# // Net ID: SXG190040
# // Natural Language Processing  (CS 6320.501) Assignment 3 (Question 2 - Viterbi Decoding Algorithm) -->

# Implementation of Viterbi Decoding(HMM) algorithm
import numpy as np


def Viterbi_Decoding(Transition_Probs, Start_Trans_Prob, Word_Tag_Probs, Observ_Sequence):
    States_Tag_Count = Transition_Probs.shape[0]  # Number of states
    Sequence_Count = Observ_Sequence.shape[1]  # length of Sentence observation sequence

    # Initializing Viterbi_Matrix and MAX_IDX_Matrix matrices
    Viterbi_Matrix = np.zeros([States_Tag_Count, Sequence_Count]) # 7X5
    MAX_IDX_Matrix = np.zeros([States_Tag_Count, Sequence_Count - 1]) # 7X4
    Viterbi_Matrix[:, 0] = np.multiply(Start_Trans_Prob, Word_Tag_Probs[:, 0]) # 7X5


    # Calculting Viterbi Matrix Probs
    for n in range(1, Sequence_Count):
        for i in range(States_Tag_Count):
            # transition * prob(OBS|State)
            Current_product = np.multiply(Transition_Probs[:, i], Viterbi_Matrix[:, n - 1])
            Viterbi_Matrix[i, n] = np.amax(Current_product* Word_Tag_Probs[i, Observ_Sequence[0, n] - 1])
            # Maximum position at Current_product* Word_Tag_Probs[i, Observ_Sequence[0, n] - 1]
            MAX_IDX_Matrix[i, n - 1] = np.argmax(Current_product* Word_Tag_Probs[i, Observ_Sequence[0, n] - 1])



    MAX_POS = np.zeros([1, Sequence_Count])
    MAX_POS[0, -1] = np.argmax(Viterbi_Matrix[:, -1])

    # Backtracking for Getting the TAGS
    for k in range(Sequence_Count - 2, -1, -1):
        MAX_POS[0, k] = MAX_IDX_Matrix[int(MAX_POS[0, k + 1]), k]


    # Converting the indices to State indices
    TAGS_IDX = MAX_POS.astype(int) + 1

    # Calculating the Probabilities at Each State by Backtracking
    MAX_POS = MAX_POS.astype(int)
    TAGS_PROBS = []
    for i, j in enumerate(MAX_POS[0]):
        # print(Viterbi_Matrix[j, i])
        TAGS_PROBS.append(Viterbi_Matrix[j, i])
    # print("TAGS_PROBS", TAGS_PROBS)


    return TAGS_IDX,TAGS_PROBS




# Main Function
if __name__ == "__main__":

    #  Initialling the Transition Probabilities
    Transition_Probs = np.array([[0.3777,0.0110,0.0009,0.0084,0.0584,0.0090,0.0025],
                  [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041],
                  [0.0322,0.0005,0.0050,0.0837,0.0615,0.0514,0.2231],
                  [0.0366,0.0004,0.0001,0.0733,0.4509,0.0036,0.0036],
                  [0.0096,0.0176,0.0014,0.0086,0.1216,0.0177,0.0068],
                  [0.0068,0.0102,0.1011,0.1012,0.0120,0.0728,0.0479],
                  [0.1147,0.0021,0.0002,0.2157,0.4744,0.0102,0.0017]])

    #   Probabilities of Word given tags
    Word_Tag_Probs = np.array([[0.000032,0.0,0.0,0.000048,0.0],
                  [0.0,0.308431,0.0,0.0,0.0],
                  [0.0,0.000028,0.000672,0.0,0.000028],
                  [0.0,0.0,0.000340,0.0,0.0],
                  [0.0,0.000200,0.000223,0.0,0.002337],
                  [0.0,0.0,0.010446,0.0,0.0],
                  [0.0,0.0,0.0,0.506099,0.0]])
    #   Probabilities of transition states from the start Tag
    Start_Trans_Prob = np.array([[0.2767, 0.0006, 0.0031,0.0453,0.0449,0.0510,0.2026]])



    # Sentence Observations
    Sentence_SEQ_DICT={
      "janet":1,
      "will":2,
      "back":3,
      "the":4,
      "bill":5
    }

    # Tagging States
    TAGS_DICT={
          "1":"NNP",
          "2":"MD",
          "3":"VB",
          "4":"JJ",
          "5":"NN",
          "6":"RB",
          "7":"DT"
    }


    # TEST_Sentence=input("Please Enter a sentance for Vitebi HMM Decoding: \n")
    # TEST_Sentence="Janet will back the bill"
    # TEST_Sentence = "will Janet back the bill"
    TEST_Sentence = "back the bill Janet will"
    TEST_Sentence=TEST_Sentence.split()
    Sentence_SEQ_Words=[[]]
    for token in TEST_Sentence:
        Sentence_SEQ_Words[0].append(Sentence_SEQ_DICT.get(token.lower()))

    Observ_Sequence = np.stack(np.array(Sentence_SEQ_Words))
    # print(Observ_Sequence)

    # Calling Viterbi Decoding 
    TAGS_IDX,TAGS_PROBS = Viterbi_Decoding(Transition_Probs, Start_Trans_Prob, Word_Tag_Probs, Observ_Sequence)

    print(" \n \n ===== Viterbi HMM Decoding for Most Likely POS Tag Sequence ============")
    print('\n Observation sequence for the Sentence:', str(TEST_Sentence))
    print("=============================================")

    MOST_Likely_TAG_SEQ=[]
    for t in TAGS_IDX[0]:
        MOST_Likely_TAG_SEQ.append(TAGS_DICT.get(str(t)))

    print("Most Likely Tag Sequence:  "+str(MOST_Likely_TAG_SEQ))
    print("----------------------------------------")

    print("PROBABILTY of the TAG Sequence at each State/TAG is :"+str(TAGS_PROBS))
    print("----------------------------------------")

    print("Overall Probability of the Tag Sequence is:" + str(TAGS_PROBS[-1]))
    print("=============================================")