
#draft from question 4 in house price

# p_ls = [] from here down
    # mean_ls = []
    # std_ls = []
    # for p in range(10,101):
    #     loss_sum = 0
    #     loss_ls = []
    #     for _ in range(10):
    #         #10 samples all the time,
    #         x_train, y_train, x_test, y_test = split_train_test(X, Y, p/100)
    #         reg.fit(x_train.to_numpy(), y_train.to_numpy())
    #         loss = reg.loss(x_test.to_numpy(), y_test.to_numpy())
    #         loss_ls.append(loss)
    #     p_ls.append(p)
    #     mean_ls.append(np.mean(loss_ls))
    #     std_ls.append(np.std(loss_ls))
    #     # creating the averege loss
    # arr1 = np.array(p_ls)
    # arr2 = np.array(mean_ls)
    # arr3 = np.array(std_ls)
    # fig2 = go.Figure(go.Scatter(x=arr1, y=arr2, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
    #                  )
    # fig2.show()