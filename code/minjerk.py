def minjerk(H1x,H1y,H2x,H2y,t,n):
	"""
	Given hand initial position H1x,H1y, final position H2x,H2y and movement duration t,
	and the total number of desired sampled points n,
	Calculates the hand path H over time T that satisfies minimum-jerk.
	
	Flash, Tamar, and Neville Hogan. "The coordination of arm
        movements: an experimentally confirmed mathematical model." The
        journal of Neuroscience 5, no. 7 (1985): 1688-1703.
	
	"""
	T = linspace(0,t,n)
	Hx = zeros(n)
	Hy = zeros(n)
	for i in range(n):
		tau = T[i]/t
		Hx[i] = H1x + ((H1x-H2x)*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
		Hy[i] = H1y + ((H1y-H2y)*(15*(tau**4) - (6*tau**5) - (10*tau**3)))
	return Hx,Hy

