library(cpm)

d <- 20
change_at <- 500
ones = matrix(rep(1,d), nrow = d, ncol = d)

# run the covariance shift experiments
for (run in 1:20) {
    pre_change <- MASS::mvrnorm(change_at, mu=rep(0,d), Sigma=diag(d))
    post_change <- MASS::mvrnorm(100, mu=rep(0,d), Sigma=ones)
    data <- rbind(pre_change,post_change)

    acc <- c()
    for (j in 1:d) {
        cpm <- makeChangePointModel(cpmType="Cramer-von-Mises", ARL0=50000)

        for (i in 1:length(data[,1])) {
            cpm <- processObservation(cpm,data[i,j])
        }

        acc <- rbind(acc,getStatistics(cpm))
    }
    write.table(acc, paste("results/R/cov_shift/CvM/out", run, ".csv", sep=""))
}


# run the mean shift experiments
for (run in 1:20) {
    pre_change <- MASS::mvrnorm(change_at, mu=rep(0,d), Sigma=diag(d))
    post_change <- MASS::mvrnorm(100, mu=rep(1,d), Sigma=diag(d))
    data <- rbind(pre_change,post_change)

    acc <- c()
    for (j in 1:d) {
        cpm <- makeChangePointModel(cpmType="Cramer-von-Mises", ARL0=50000)

        for (i in 1:length(data[,1])) {
            cpm <- processObservation(cpm,data[i,j])
        }

        acc <- rbind(acc,getStatistics(cpm))
    }
    write.table(acc, paste("results/R/mean_shift/CvM/out", run, ".csv", sep=""))
}
