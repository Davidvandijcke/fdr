library(bootstrap)
X <- read.table(url("http://www.stat.umn.edu/geyer/5601/mydata/big-unif.txt"),
               header = TRUE)
attach(X)

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
stripchart(lqlist, xlab = "subsample size",
           ylab = "log(high quantile - low quantile)",
           vertical = TRUE)

y <- sapply(lqlist, mean)
print(beta <- cov(- y, log(b)) / var(log(b)))

# confidence interval calculation
m <- 3
b <- b[m]
theta.star <- theta.star[ , m]
alpha <- 0.05
z.star <- b^beta * (theta.star - theta.hat)

# two-sided interval
crit.val <- sort(z.star)[(nboot + 1) * c(1 - alpha / 2, alpha / 2)]
theta.hat - crit.val / n^beta

# histogram from which critical values are derived
hist(z.star)
abline(v = crit.val, lty = 2)

# one-sided confidence interval actually makes more sense
# since we know theta.hat < theta
crit.val <- sort(z.star)[(nboot + 1) * alpha]
theta.hat - c(0, crit.val) / n^beta

cat("Calculation took", proc.time()[1], "seconds\n")