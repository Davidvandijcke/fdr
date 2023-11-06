library(bootstrap)
X <- read.table(url("http://www.stat.umn.edu/geyer/5601/mydata/big-unif.txt"),
               header = TRUE)
attach(X)

# estimating the rate
print(n <- length(x))
print(theta.hat <- max(x))

nboot <- 2e4 - 1
b <- c(40, 60, 90, 135)
b <- sort(b)
theta.star <- matrix(NA, nboot, length(b))
for (i in 1:nboot) {
  x.star <- x
  for (j in length(b):1) {
    x.star <- sample(x.star, b[j], replace = FALSE)
    theta.star[i, j] <- max(x.star)
  }
}

zlist <- list()
for (i in 1:length(b)) {
  zlist[[i]] <- theta.star[ , i] - theta.hat
}
names(zlist) <- b
boxplot(zlist, xlab = "subsample size",
        ylab = expression(hat(theta)[b] - hat(theta)[n]))

qlist <- list()
k <- (nboot + 1) * seq(0.05, 0.45, 0.05)
l <- (nboot + 1) * seq(0.55, 0.95, 0.05)
for (i in 1:length(b)) {
  z.star <- zlist[[i]]
  sz.star <- sort(z.star, partial = c(k, l))
  qlist[[i]] <- sz.star[l] - sz.star[k]
}
names(qlist) <- b

lqlist <- lapply(qlist, log)


y <- sapply(lqlist, mean)

