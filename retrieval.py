# retrieval results (testing)
import numpy as np
import utilities as ut
import pandas as pd

_RESULTS_ = 15

# for one query
def eva(query_x, query_id_y, database_x, database_id_y, top = False):

    distance = ut.cosine_distance(query_x,database_x).reshape((-1,1))

    result = np.hstack((distance, database_id_y))
    result = result[result[:, 0].argsort()] # rank with distance
    result_all = np.delete(result, [0,1], axis=1).reshape(-1) # get labels

    if top:
        result_count_1 = result_all[range(1)]  # Top 1
        result_count_10 = result_all[range(10)] # Top 10

        precision_1 = 1 - np.count_nonzero(result_count_1-query_id_y[1])
        precision_10 = 1 - np.count_nonzero(result_count_10-query_id_y[1]) / 10
    else:
        precision_1 = 0
        precision_10 = 0
    # AP
    reAP = result_all - query_id_y[1]
    P = 0.
    acc = 0
    last = reAP.shape[0] - np.count_nonzero(database_id_y[:,1]-query_id_y[1])
    for i in range(reAP.shape[0]):
        if acc == last:
            break
        if reAP[i]==0:
            acc +=1
            P += acc/(i+1)
    AP = P/last

    return AP, result[range(_RESULTS_)],precision_1, precision_10

# for all queries
def get_result(query_x, query_id_y, database_x, database_id_y, querys, top = False):

    print('evaluating...')
    precision_1 = 0.
    precision_10 = 0.
    mAP = 0.
    results = []

    for i in range(querys):
        i_AP, i_result, i_p1, i_p10 = eva(query_x[i], query_id_y[i], database_x, database_id_y, top)
        precision_1 += i_p1
        precision_10 += i_p10
        mAP += i_AP
        one_result = [query_x[i,0], i_AP, i_p1, i_p10, i_result[:,1].tolist()] # id,AP,p1,p10,result_ids
        results.append(one_result) # get id only
        if i % 100 == 0:
            print('--%d--' % i)
    precision_1 /= querys
    precision_10 /= querys
    mAP /= querys

    return mAP,results,precision_1,precision_10

def main(top = False, save = False):
    database_feature = np.loadtxt('csv/database.csv')
    database_id_label = np.loadtxt('csv/database_index.csv', dtype=int, delimiter=',')
    query_feature = np.loadtxt('csv/query.csv')
    query_id_label = np.loadtxt('csv/query_index.csv', dtype=int, delimiter=',')
    querys = query_feature.shape[0]

    mAP, results, precision_1, precision_10 = get_result(
        query_feature,query_id_label,database_feature,database_id_label,querys,top=top)

    if top:
        print('mAP:%.6f, P@1:%.6f, P@10:%.6f' % (mAP,precision_1,precision_10))
    else:
        print('mAP:%.6f' % mAP)

    if save:
        results = pd.DataFrame(results)
        results.to_csv('csv/retrieval_results.csv',header=['id','AP','p1','p10','result_ids'],index=False)


if __name__ == '__main__':
    main(top=False, save=True)
