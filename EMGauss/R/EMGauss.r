#Fonction de base
rnormmix = function(n, theta){
  K = length(theta$pi)
  k = sample.int(K, n, replace = TRUE, prob = theta$pi)
  simulation = rnorm(n, theta$mu[k], theta$sigma[k])
  return(simulation)
}

dnormmix = function(x, theta){
  dens = sapply(x, function(y) sum(theta$pi*dnorm(y,theta$mu, theta$sigma)))
  return(dens)
}

logvrainorm = function(param, obs){
  return(sum(log(dnormmix(obs,param))))
}

#Start EM
update.theta = function(obs,theta){
  K = length(theta$pi)
  pi.new = numeric(K)
  mu.new = numeric(K)
  sigma.new = numeric(K)
  
  for (k in 1:K){
    alpha = theta$pi[k]*dnorm(obs,theta$mu[k], theta$sigma[k])/dnormmix(obs,theta)
    pi.new[k] = mean(alpha)
    mu.new[k] = sum(obs*alpha)/(length(obs)*pi.new[k])
    vec.var = sapply(obs, function(x) (x-mu.new[k])^2)
    sigma.new[k] = sqrt(sum(alpha*vec.var)/(length(obs)*pi.new[k]))
  }
  
  return(list(pi = pi.new, mu = mu.new, sigma = sigma.new))
}

#If there exists an initial guess
algoEM = function(obs, theta_init){
  eps = 10^-3
  count = 0
  new_theta = update.theta(obs,theta_init)
  while ((abs(logvrainorm(new_theta, obs)/logvrainorm(theta_init, obs) - 1) >= eps) & count <= 200){
    theta_init = new_theta
    new_theta = update.theta(obs, theta_init)
    count = count + 1
  }
  return(list(emv = new_theta, no_iter = count))
}

#If no initial guess possible
algoEM_update = function(obs, K){
  mu = sample(obs, K)
  sig = replicate(K, sd(obs))
  pi = replicate(K,1)/K
  theta_init = list(pi = pi, mu = mu, sigma = sig)
  
  eps = 10^-3
  count = 0
  
  new_theta = update.theta(obs,theta_init)
  while ((abs(logvrainorm(new_theta, obs)/logvrainorm(theta_init, obs) - 1) >= eps) & count <= 200){
    theta_init = new_theta
    new_theta = update.theta(obs, theta_init)
    count = count + 1
  }
  return(list(emv = new_theta, no_iter = count))
}

#return theta with max log likelihood
EMGauss = function(obs, K, L){
  log_EM = numeric(L)
  x = list()
  for(l in 1:L){
    em = algoEM_update(obs,K)$emv
    log_EM[l] = logvrainorm(em, obs)
    x[[l]] = em
  }
  index = which.max(log_EM)
  return(x[[index]])
}