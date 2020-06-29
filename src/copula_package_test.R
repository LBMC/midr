library(copula)

x <- matrix(c(0.42873569, 0.18285458, 0.9514195,
0.25148149, 0.05617784, 0.3378213,
0.79410993, 0.76175687, 0.0709562,
0.02694249, 0.45788802, 0.6299574,
0.39522060, 0.02189511, 0.6332237,
0.66878367, 0.38075101, 0.5185625,
0.90365653, 0.19654621, 0.6809525,
0.28607729, 0.82713755, 0.7686878,
0.22437343, 0.16907646, 0.5740400,
0.66752741, 0.69487362, 0.3329266), ncol=3, byrow=T)

copClayton@dacopula(x, 1.2, log=F)
copClayton@dacopula(x, 1.2, log=T)

copClayton@iPsi(x, 0.2, log=F)
copClayton@iPsi(x, 0.2, log=T)
copClayton@psi(x, 0.2)
copClayton@dDiag(apply(x, 1, max), 0.2, 3, log=F)
copClayton@dDiag(apply(x, 1, max), 0.2, 3, log=T)

copFrank@psi(x, 0.2)
copFrank@psi(x, -37)
copFrank@psi(x, -10)
copFrank@iPsi(x, 0.2, log=F)
copFrank@dDiag(apply(x, 1, max), 0.2, 3, log=F)
copFrank@dacopula(x, 5.0, log=F, Li.log.arg = F)
copFrank@dacopula(x, 5.0, log=T, Li.log.arg = T)

polylog(c(0.01556112, 0.00108968, 0.00889932), -2, method = "negI-s-Eulerian", log=T)
polylog(log(c(0.01556112, 0.00108968, 0.00889932)), -2, method = "negI-s-Eulerian", log=T, is.log.z = T)

copGumbel@iPsi(x, 1.2, log=F)
copGumbel@iPsi(x, 1.2, log=T)

copGumbel@psi(x, 1.2)
copGumbel@iPsi(x, 1.2, log=T)
copGumbel@dDiag(apply(x, 1, max), 0.2, 3, log=F)
copGumbel@dDiag(apply(x, 1, max), 0.2, 3, log=T)
copGumbel@dacopula(x, 1.2, log=F)
copGumbel@dacopula(x, 1.2, log=T)
copGumbel@dacopula(x[, 1:2], 3.2, log=T)

polyG <- function(lx, alpha, d, method= c("default", "default2012", "default2011",
                                          "pois", "pois.direct", "stirling", "stirling.horner",
                                          coeffG.methods),
                  verboseUsingRmpfr = isTRUE(getOption("copula:verboseUsingRmpfr")),
                  log=FALSE)
{
  stopifnot(length(alpha) == 1L, 0 < alpha, alpha <= 1,
            d == as.integer(d), d >= 1)
  k <- 1:d
  Meth <- method
  switch(Meth,
         "default" =, "default2012" =
           {
             ## "default2012" compiled by Yongsheng Wang (MSc thesis c/o M.Maechler, April 2012)
             ## it switches to "Rmpfr" when the accuracy would be less than 5 digits
             meth2012 <- function(d, alpha, lx) {
               if (d <= 30) "direct"
               else if (d <= 50) {
                 if (alpha <= 0.8) "direct" else "dsSib.log"
               }
               else if (d <= 70) {
                 if (alpha <= 0.7) "direct" else "dsSib.log"
               }
               else if (d <= 90) {
                 if (alpha <= 0.5) "direct"
                 else if (alpha >= 0.8) "dsSib.log"
                 else if (lx <= 4.08) "pois"
                 else if (lx >= 5.4) "direct"
                 else "dsSib.Rmpfr"
               }
               else if (d <= 120) {
                 if (alpha < 0.003) "sort"
                 else if (alpha <= 0.4) "direct"
                 else if (alpha >= 0.8) "dsSib.log"
                 else if (lx <= 3.55) "pois"
                 else if (lx >= 5.92) "direct"
                 else "dsSib.Rmpfr"
               }
               else if (d <= 170) {
                 if (alpha < 0.01) "sort"
                 else	if (alpha <= 0.3) "direct"
                 else if (alpha >= 0.9) "dsSib.log"
                 else if (lx <= 3.55) "pois"
                 else "dsSib.Rmpfr"
               }
               else if (d <= 200) {
                 if (lx <= 2.56) "pois"
                 else if (alpha >= 0.9) "dsSib.log"
                 else "dsSib.Rmpfr"
               }
               else "dsSib.Rmpfr"
             }
             ix <- seq_along(lx)
             ## Each lx can -- in principle -- ask for another method ... --> split() by method
             meth.lx <- tryCatch(## when lx is "mpfr", vapply() currently (2016-08) fails
               vapply(lx, function(lx) meth2012(d, alpha, lx), ""),
               error = { ch <- character(length(lx))
               for (i in ix) ch[i] <- meth2012(d, alpha, lx[[i]]); ch })
             if(verboseUsingRmpfr && (lg <- length(grep("Rmpfr$", meth.lx))))
               message("Default method chose 'Rmpfr' ", if(lg > 1) paste(lg,"times") else "once")
             i.m <- split(ix, factor(meth.lx))
             r <- lapply(names(i.m), function(meth)
               polyG(lx[i.m[[meth]]], alpha = alpha, d = d, method = meth, log = log))
             lx[unlist(i.m, use.names=FALSE)] <- unlist(r, use.names=FALSE)
             lx
           },
         
         "default2011" = ## first "old" default
           {
             Recall(lx, alpha=alpha, d=d,
                    method = if(d <= 100) {
                      if(alpha <= 0.54) "stirling"
                      else if(alpha <= 0.77) "pois.direct"
                      else "dsSib.log"
                    } else "pois", # slower but more stable, e.g., for d=150
                    log=log)
           },
         "pois" =
           {
             ## build list of b's
             n <- length(lx)
             x <- exp(lx)                                   # e^lx = x
             lppois <- outer(d-k, x, FUN=ppois, log.p=TRUE) # a (d x n)-matrix; log(ppois(d-k, x))
             print("lppois")
             print(lppois)
             llx <- k %*% t(lx)           # also a (d x n)-matrix; k*lx
             print("llx")
             print(llx)
             labsPoch <- vapply(k, function(j) sum(log(abs(alpha*j-(k-1L)))), NA_real_) # log|(alpha*j)_d|, j=1,..,d
             print("labsPoch")
             print(labsPoch)
             lfac <- lfactorial(k)        # log(j!), j=1,..,d
             print("lfac")
             print(lfac)
             ## build matrix of exponents
             lxabs <- llx + lppois + rep(labsPoch - lfac, n) + rep(x, each = d)
             print("lxabs")
             print(lxabs)
             print("rep(labsPoch - lfac, n)")
             print(rep(labsPoch - lfac, n))
             print("rep(x, each = d)")
             print(rep(x, each = d))
             res <- lssum(lxabs, signFF(alpha, k, d), strict=FALSE)
             print(res)
             if(log) res else exp(res)
           },
         "pois.direct" =
           {
             ## build coefficients
             xfree <- lchoose(alpha*k,d) + lfactorial(d) - lfactorial(k)
             x <- exp(lx)
             lppois <- outer(d-k, x, FUN=ppois, log.p=TRUE) # (length(x),d)-matrix
             klx <- lx %*% t(k)
             exponents <- exp(t(x+klx)+lppois+xfree) # (d,length(x))-matrix
             res <- as.vector(signFF(alpha, k, d) %*% exponents)
             if(log) log(res) else res
           },
         "stirling" =
           {
             ## implementation of \sum_{k=1}^d a_{dk}(\theta) x^k
             ## = (-1)^{d-1} * x * \sum_{k=1}^d alpha^k * s(d,k) * \sum_{j=1}^k S(k,j) * (-x)^{j-1}
             ## = (-1)^{d-1} * x * \sum_{k=1}^d alpha^k * s(d,k) * polynEval(...)
             ## inner function is evaluated via polynEval
             x <- exp(lx)
             s <- Stirling1.all(d) # s(d,1), ..., s(d,d)
             S <- lapply(k, Stirling2.all) # S[[l]][n] contains S(l,n), n = 1,...,l
             lst <- lapply(k, function(k.) (-1)^(d-1)*x*alpha^k.*s[k.]*polynEval(S[[k.]],-x))
             res <- rowSums(matrix(unlist(lst), nrow=length(x)))
             if(log) log(res) else res
           },
         "stirling.horner" =
           {
             ## implementation of \sum_{k=1}^d a_{dk}(\theta) x^k
             ## = (-1)^{d-1} * x * \sum_{k=1}^d alpha^k * s(d,k) * \sum_{j=1}^k S(k,j) * (-x)^{j-1}
             ## = (-1)^{d-1} * x * \sum_{k=1}^d alpha^k * s(d,k) * polynEval(...)
             ## polynEval is used twice
             x <- exp(lx)
             s <- Stirling1.all(d) # s(d,1), ..., s(d,d)
             S <- lapply(k, Stirling2.all) # S[[l]][n] contains S(l,n), n = 1,...,l
             len <- length(x)
             poly <- matrix(unlist(lapply(k, function(k.) polynEval(S[[k.]],-x))), nrow=len) # (len,d)-matrix
             res <- (-1)^(d-1)*alpha*x* vapply(1:len, function(i) polynEval(s*poly[i,], alpha), 1.)
             if(log) log(res) else res
             ## the following code was *not* faster
             ## poly <- t(sapply(k, function(k.) polynEval(S[[k.]],-x))) # (d,len(x))-matrix
             ## coeff <- if(length(x)==1) t(s*poly) else s*poly
             ## res <- (-1)^(d-1)*alpha*x*apply(coeff, 2, polynEval, x=alpha)
           },
         "coeffG" = ## <<< all the different 'coeffG' methods --------------------
         {
           ## note: these methods are all known to show numerical deficiencies
           if(d > 220) stop("d > 220 not yet supported") # would need Stirling2.all(d, log=TRUE)
           ## compute the log of the coefficients:
           l.a.dk <- coeffG(d, alpha, method=method, log = TRUE)
           ##	     ~~~~~~			     ~~~~~~~~~~
           ## evaluate the sum
           ## for this, create a matrix B with (k,i)-th entry
           ## B[k,i] = log(a_{dk}(theta)) + k * lx[i],
           ##          where k in {1,..,d}, i in {1,..,n} [n = length(lx)]
           logx <- l.a.dk + k %*% t(lx)
           if(log) {
             ## compute log(colSums(exp(B))) stably (no overflow) with the idea of
             ## pulling out the maxima
             lsum(logx)
           } else colSums(exp(logx))
         },
         stop(gettextf("unsupported method '%s' in polyG", method), domain=NA)
  ) # end{switch}
}## {polyG}
lsum <- function(lx, l.off) {
  rx <- length(d <- dim(lx))
  if(mis.off <- missing(l.off)) l.off <- {
    if(rx <= 1L)
      max(lx)
    else if(rx == 2L)
      apply(lx, 2L, max)
  }
  if(rx <= 1L) { ## vector
    if(is.finite(l.off))
      l.off + log(sum(exp(lx - l.off)))
    else if(mis.off || is.na(l.off) || l.off == max(lx))
      l.off # NA || NaN or all lx == -Inf, or max(.) == Inf
    else
      stop("'l.off  is infinite but not == max(.)")
  } else if(rx == 2L) { ## matrix
    if(any(x.off <- !is.finite(l.off))) {
      if(mis.off || isTRUE(all.equal(l.off, apply(lx, 2L, max)))) {
        ## we know l.off = colMax(.)
        if(all(x.off)) return(l.off)
        r <- l.off
        iok <- which(!x.off)
        l.of <- l.off[iok]
        r[iok] <- l.of + log(colSums(exp(lx[,iok,drop=FALSE] -
                                           rep(l.of, each=d[1]))))
        r
      } else ## explicitly specified l.off differing from colMax(.)
        stop("'l.off' has non-finite values but differs from default max(.)")
    }
    else
      l.off + log(colSums(exp(lx - rep(l.off, each=d[1]))))
  } else stop("not yet implemented for arrays of rank >= 3")
}
lssum <- function (lxabs, signs, l.off = apply(lxabs, 2, max), strict = TRUE) {
  stopifnot(length(dim(lxabs)) == 2L) # is.matrix(.) generalized
  print("l.off, each=nrow(lxabs)")
  print(l.off, each=nrow(lxabs))
  print("lxabs - rep(l.off, each=nrow(lxabs))")
  print(lxabs - rep(l.off, each=nrow(lxabs)))
  print("exp(lxabs - rep(l.off, each=nrow(lxabs)))")
  print(exp(lxabs - rep(l.off, each=nrow(lxabs))))
  print("signs")
  print(signs)
  print("signs * exp(lxabs - rep(l.off, each=nrow(lxabs)))")
  print(signs * exp(lxabs - rep(l.off, each=nrow(lxabs))))
  sum. <- colSums(signs * exp(lxabs - rep(l.off, each=nrow(lxabs))))
  if(anyNA(sum.) || any(sum. <= 0))
    (if(strict) stop else warning)("lssum found non-positive sums")
  l.off + log(sum.)
}
signFF <- function(alpha, j, d) {
  stopifnot(0 < alpha, alpha <= 1, d >= 0, 0 <= j)
  res <- numeric(length(j))
  if(alpha == 1) {
    res[j == d] <- 1
    res[j > d] <- (-1)^(d-j)
  } else {
    res[j > d] <- NA # the formula below does not hold {TODO: find correct sign}
    ## we do not need them in dsumSibuya() and other places...
    x <- alpha*j
    ind <- x != floor(x)
    res[ind] <- (-1)^(j[ind]-ceiling(x[ind]))
  }
  res
}


polyG <- function(lx, alpha, d, method= c("default", "default2012", "default2011",
                                          "pois", "pois.direct", "stirling", "stirling.horner",
                                          coeffG.methods),
                  verboseUsingRmpfr = isTRUE(getOption("copula:verboseUsingRmpfr")),
                  log=FALSE)
{
  k <- 1:d
  print(k)
  ## build coefficients
  print(alpha * k)
  print(d)
  xfree <- lchoose(alpha*k,d) + lfactorial(d) - lfactorial(k)
  print(lchoose(alpha*k,d))
  print(combn(alpha*k,d))
  print(lfactorial(d))
  print(lfactorial(k))
  print(xfree)
  x <- exp(lx)
  lppois <- outer(d-k, x, FUN=ppois, log.p=TRUE) # (length(x),d)-matrix
  klx <- lx %*% t(k)
  exponents <- exp(t(x+klx)+lppois+xfree) # (d,length(x))-matrix
  res <- as.vector(signFF(alpha, k, d) %*% exponents)
  if(log) log(res) else res
}
polyG(0.8386414, 1/1.2, 3, method = "pois.direct", log=T)
?lchoose

lsum( copGumbel@iPsi(x[1,], 1.2, log=T))
polyG(lsum( copGumbel@iPsi(x[1,], 1.2, log=T)) * 1/1.2, 1/1.2, 3, method = "pois", log=T)
polyG(lsum( copGumbel@iPsi(x[2,], 1.2, log=T)) * 1/1.2, 1/1.2, 3, method = "pois", log=T)
polyG(lsum( copGumbel@iPsi(x[3,], 1.2, log=T)) * 1/1.2, 1/1.2, 3, method = "pois", log=T)
polyG(lsum( copGumbel@iPsi(x, 1.2, log=T)) * 1/1.2, 1/1.2, 3, method = "pois", log=T)
polyG(lsum( t(copGumbel@iPsi(x, 1.2, log=T))) * 1/1.2, 1/1.2, 3, method = "pois", log=T)
polyG(lsum( t(copGumbel@iPsi(x, 3.2, log=T))) * 1/3.2, 1/3.2, 3, method = "pois", log=T)
lsum( t(copGumbel@iPsi(x, 1.2, log=T)))
t(copGumbel@iPsi(x, 1.2, log=T))

signFF(1/1.2, 1, 3)
signFF(1/1.2, 2, 3)
signFF(1/1.2, 1:3, 3)
signFF(1/3.2, 1:3, 3)
signFF(1/30.2, 1:3, 3)

log(sum( copGumbel@iPsi(x[1,], 1.2, log=F))) * 1/1.2

lsum(t(copGumbel@iPsi(x, 1.2, log=T)))

lssum(x, signFF(1/3.2, 1:3, 3))
lssum(t(x), signFF(1/3.2, 1:3, 3))

polyG(0.8386414, 1/1.2, 3, method = "pois", log=T)
polyG(0.8386414, 1/1.2, 3, method = "pois.direct", log=T)

k <- 1:3
alpha <- 1 / 1.2
vapply(k, function(j) sum(log(abs(alpha*j-(k-1L)))), NA_real_)

copula::polyG

.Machine$double.eps

copFrank@dacopula(x, 0.2, log=F, Li.log.arg = F) - c(0.94796045, 1.07458178, 0.91117583, 0.98067912, 0.99144689, 0.9939432 , 0.94162409, 0.96927238, 1.02271257, 0.98591624)
copFrank@dacopula(x, 0.2, log=F, Li.log.arg = F) - c(0.94796044, 1.0745818 , 0.91117584, 0.98067908, 0.99144693,
0.99394317, 0.94162409, 0.96927238, 1.02271253, 0.98591628)

polylog(log(c(0.01556112, 0.00108968, 0.00889932)), -2, is.log.z = TRUE, log = T)
polylog(c(0.01556112, 0.00108968, 0.00889932), -2, is.log.z = TRUE, log = F)
