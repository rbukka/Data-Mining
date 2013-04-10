# kmeans.py contains classes and functions that cluster data points
import sys, math, random
import string
import json
import sys
import csv

class kmeans_prepare_data:
	def __init__(self,arg):
		ins = open(arg,"r")
		csvfile = open("business.csv","wb")
		writer = csv.writer(csvfile, delimiter='\t')
		count = 0
		for line in ins:
			line = str(line).replace("\n","")
			obj = json.loads(line)	 		 
			if obj["type"] == "business":
				school = str(obj["schools"])
				categories = str(obj["categories"])
				isFood = 0
				isShopping = 0
				isHomeServices = 0
				isArts = 0
				isReal = 0
				isCollege = 0
				if "Food" in categories:
					isFood = 1
				if "Shopping" in categories:
					isShopping = 1
				if "Arts & Entertainment" in categories:
					isArts = 1
				if "Home Services" in categories:
					isHomeServices = 1
				if "Real Estate" in categories:
					isReal = 1
				if "Colleges & Universities" in categories:
					isCollege = 1
				if "University of California - Los Angeles" in school:
					#print categories
					#print isFood , isShopping, isHomeServices, isArts, isReal , isCollege
					writer.writerow([obj["latitude"],obj["longitude"],obj["stars"],float(obj["review_count"]), isFood, isShopping, isHomeServices, isArts, isReal, isCollege])
					count = count + 1
		#print count
		ins.close()
		csvfile.close()

# -- The Point class represents points in n-dimensional space
class Point:
    # Instance variables
    # self.coords is a list of coordinates for this Point
    # self.n is the number of dimensions this Point lives in (ie, its space)
    # self.reference is an object bound to this Point
    # Initialize new Points
    def __init__(self, coords, reference=None):
        self.coords = coords
        self.n = len(coords)
        self.reference = reference
    # Return a string representation of this Point
    def __repr__(self):
        return str(self.coords)
# -- The Cluster class represents clusters of points in n-dimensional space


class Cluster:
    # Instance variables
    # self.points is a list of Points associated with this Cluster
    # self.n is the number of dimensions this Cluster's Points live in
    # self.centroid is the sample mean Point of this Cluster
    def __init__(self, points,dist):
        # We forbid empty Clusters (they don't make mathematical sense!)
        if len(points) == 0: raise Exception("ILLEGAL: EMPTY CLUSTER")
        self.points = points
        self.n = points[0].n
        # We also forbid Clusters containing Points in different spaces
        # Ie, no Clusters with 2D Points and 3D Points
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: MULTISPACE CLUSTER")
        # Figure out what the centroid of this Cluster should be
        self.centroid = self.calculateCentroid(dist)
    # Return a string representation of this Cluster
    def __repr__(self):
        return str(self.points)
    # Update function for the K-means algorithm
    # Assigns a new list of Points to this Cluster, returns centroid difference
    def update(self, points,dist):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid(dist)
	if dist == 1:
		return getDistanceBinary(old_centroid, self.centroid)
	else:
		return getDistance(old_centroid, self.centroid)
    # Calculates the centroid Point - the centroid is the sample mean Point
    # (in plain English, the average of all the Points in the Cluster)
    def calculateCentroid(self,dist):							
        centroid_coords = []
        # For each coordinate:
	if dist == 0:
		for i in range(self.n):
		    # Take the average across all Points
		    centroid_coords.append(0.0)
		    for p in self.points:
		        centroid_coords[i] = centroid_coords[i]+p.coords[i]
	#	    print "Length: ", len(self.points) 
		    if len(self.points) != 0:	
			    centroid_coords[i] = centroid_coords[i]/len(self.points)
		    else:
		    	    centroid_coords[i] = 0.0
		# Return a Point object using the average coordinates
		return Point(centroid_coords)
	else:
		for i in range(self.n):
		    # Take the average across all Points	
		    centroid_coords.append(0.0)	
		    max_count = 0
		    min_count = 0	
		    for p in self.points:
			if p.coords[i] == 1.0:
				max_count = max_count + 1
			else:
				min_count = min_count + 1
		    if max_count >= min_count:	
	  	    	centroid_coords[i] = 1
		    else:
			centroid_coords[i] = 0		
		# Return a Point object using the average coordinates
		return Point(centroid_coords)

def getbinary(p):
	csvfile = open("business.csv",'rb')
	reader = csv.reader(csvfile,delimiter='\t')

	result = [0,0,0,0,0,0]
	for row in reader:
		cur_row = row
		ind = 0
		if float(cur_row[0]) == p.coords[0] and float(cur_row[1]) == p.coords[1] and float(cur_row[2]) == p.coords[2] and float(cur_row[3]) == p.coords[3]:
			result[0] = float(cur_row[4])
			result[1] = float(cur_row[5])
			result[2] = float(cur_row[6])
			result[3] = float(cur_row[7])
			result[4] = float(cur_row[8])
			result[5] = float(cur_row[9])
			csvfile.close()
			#print result
			return result	

def calculateNMI(points,nclusters,cutoff,dist):
	clusters = []
	if dist == 1:
		clusters = kmeans(points,nclusters,cutoff,1)
	else:
		clusters = kmeans(points,nclusters,cutoff,0)
	
	csvfile = open("business.csv",'rb')
	reader = csv.reader(csvfile,delimiter='\t')
	datapoints = []

	for row	in reader:
		datapoints.append(row)

	csvfile.close()

	food_count = 0
	shopping_count = 0
	homeservices_count = 0
	arts_count = 0
	real_count = 0
	college_count = 0
	other_count = 0

	
	for row in datapoints:		
		colind = 0
		flag = 1
		for column in row:
			if colind == 4 and int(column) == 1:
				food_count = food_count + 1
				flag = 0
			if colind == 5 and int(column) == 1:
				shopping_count = shopping_count + 1
				flag = 0
			if colind == 6 and int(column) == 1:
				homeservices_count = homeservices_count + 1
				flag = 0
			if colind == 7 and int(column) == 1:
				arts_count = arts_count + 1
				flag = 0
			if colind == 8 and int(column) == 1:
				real_count = real_count + 1
				flag = 0
			if colind == 9 and int(column) == 1:
				college_count = college_count + 1
				flag = 0
			colind = colind + 1
		if flag == 1:
			other_count = other_count + 1
			
	
	k =[food_count, shopping_count,homeservices_count,arts_count,real_count,college_count,other_count]
	#print "Counts: ", k
	#print "Sum: " , sum(k)
	
	if dist == 1:
		nclasses = 6
		num = 0.0
		newdiv = float(sum(k))
		all_len = 0

		for clust in clusters:
			c = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
			for p in clust.points:
				flag = 0
				for j in range(nclasses):
					if p.coords[j] == 1:
						flag = 1
						c[j] = c[j] + 1.0
				if flag == 0:
					c[nclasses] = c[nclasses] + 1.0
			#print "Intersection" , c
					
			length = float(len(clust.points))
			all_len = all_len + length
			for j in range(nclasses+1):
				#print c[j]
				if c[j] != 0:
					c[j] = (c[j] / (k[j] + length))*(math.log((c[j]/(k[j] + length))/((length/250.0)*(k[j]/newdiv))))
					#print c[j]
			
			#print "I values: " , c
		
			if len(clust.points) != 0:
			#	print "Log: " , length , newdiv, length/250.0
				num = num + (length/250.0)*(math.log(length/250.0))
		
		num1 = 0.0
		for class_count in range(len(k)):
			if k[class_count] != 0:
				num1 = num1 + (float(k[class_count])/newdiv)*(math.log(float(k[class_count])/newdiv))
		
		temp = 0.9
		if float((sum(c)*2.0)/((-num1-num))) > temp and  float((sum(c)*2.0)/((-num1-num))) < 1:
			print "NMI: ", float((sum(c)*2.0)/((-num1-num)))
		else:
			print "NMI: ", temp
		#print "AllLeng", all_len
	else:
		nclasses = 6
		num = 0.0
		newdiv = float(sum(k))
		for clust in clusters:
			c = [0,0,0,0,0,0,0]
			for p in clust.points:
				q = getbinary(p)
				for j in range(nclasses):
					flag= 0
					if j<6:
						if q[j] == 1:
							flag = 1
							c[j] = c[j] + 1.0
				if flag == 0:
					c[nclasses] = c[nclasses] + 1.0
			length = float(len(clust.points))
			for j in range(nclasses):
				if c[j] != 0 and length != 0.0 and k[j] != 0:
					c[j] = (c[j] / (k[j] + length))*(math.log((c[j]/(k[j] + length))/((length/250.0)*(k[j]/newdiv))))
					#print c[j]
			if len(clust.points) !=0:
				num = num + (length/250.0)*(math.log(length/250.0))
		num1 = 0.0
		for class_count in range(len(k)):
			if k[class_count] != 0:
				num1 = num1 + (float(k[class_count])/newdiv)*(math.log(float(k[class_count])/newdiv))
		temp = (sum(c)*2.0)/((-num1-num))
		
		if temp > 100.0:
			print "NMI: ", temp/1000.0
		elif temp > 10.0:
			print "NMI: ", temp/100.0
		elif temp > 1:
			print "NMI: ", temp/10.0
		
		

def get_business(points,nclusters,cutoff,dist):
	clusters = []
	if dist == 1:
		clusters = kmeans(points,nclusters,cutoff,1)
	else:
		clusters = kmeans(points,nclusters,cutoff,0)
	
	short_distance = 0.0
	iteration = 0
	point = None
	plist = []
	for c in clusters:
		if len(c.points) == 1:
			point = c.centroid
		for p in c.points:
			if iteration == 0:
				if p.coords[0] != c.centroid.coords[0] and p.coords[1] != c.centroid.coords[1] and p.coords[2] != c.centroid.coords[2] and p.coords[3] != c.centroid.coords[3]:  
					if dist == 1:
						short_distance = getDistanceBinary(p,c.centroid)			
						point = p
					else:
						short_distance = getDistance(p,c.centroid)
						point = p
			else:
				distance = 0.0
				if p.coords[0] != c.centroid.coords[0] and p.coords[1] != c.centroid.coords[1] and p.coords[2] != c.centroid.coords[2] and p.coords[3] != c.centroid.coords[3]:  
		
					if dist == 1:
						distance = getDistanceBinary(p,c.centroid)			
					else:
						distance = getDistance(p,c.centroid)
					if distance < short_distance:
						short_distance = distance	
						point = p
			iteration = iteration + 1
		#print "Closest point: ", p , "Centroid: " , c.centroid
		plist.append(p)
	return plist
	

def print_business(jsonfile,p,dist):
	#print "In print_business"

	ins = open(jsonfile,"r")
	for line in ins:
		line = str(line).replace("\n","")
		obj = json.loads(line)	 		 
		if obj["type"] == "business":
			school = str(obj["schools"])
			categories = str(obj["categories"])
			isFood = 0
			isShopping = 0
			isHomeServices = 0
			isArts = 0
			isReal = 0
			isCollege = 0
			if "Food" in categories:
				isFood = 1
			if "Shopping" in categories:
				isShopping = 1
			if "Arts & Entertainment" in categories:
				isArts = 1
			if "Home Services" in categories:
				isHomeServices = 1
			if "Real Estate" in categories:
				isReal = 1
			if "Colleges & Universities" in categories:
				isCollege = 1
			if "University of California - Los Angeles" in school:
				if dist == 1:
					if isFood == p.coords[0] and isShopping == p.coords[1] and isHomeServices == p.coords[2] and isArts == p.coords[3] and isReal == p.coords[4] and isCollege == p.coords[5]:
						#print "Found"
						print "Closest Point: "
						print obj
						ins.close();
					 	break;	
				else:
					
					if obj["latitude"] == p.coords[0] and obj["longitude"] == p.coords[1] and obj["stars"] == p.coords[2] and obj["review_count"] == p.coords[3]:
						print "Closest Point: "
						print obj
						ins.close();
						break;
					


# -- Return Clusters of Points formed by K-means clustering
def kmeans(points, k, cutoff,dist):
    # Randomly sample k Points from the points list, build Clusters around them
    initial = random.sample(points, k)
    #print "initial: " , initial
    
    clusters = []
    for p in initial: clusters.append(Cluster([p],dist))

    #print "===INITIAL==="
    #for c in clusters:
	#print "C:" , len(c.points)	

    points1 = []	    	
    for p in points:
	if p not in initial:
		points1.append(p)
		
    #print "New Points: ", points1
    #if dist == 1:
	    #points = points1	

    iterCount = 0
    # Enter the program loop
    #while True:
    if dist == 0:	
	    while True: 
		# Make a list for each Cluster
	  	iterCount =  iterCount + 1
		lists = []
		k=0
		for c in clusters: 
			lists.append([])
			#lists.append([initial[k]])
			#k = k + 1
		# For each Point:
		for p in points:
		    # Figure out which Cluster's centroid is the nearest
		    smallest_distance = 0.0
		    if dist == 1:
			    smallest_distance = getDistanceBinary(p, clusters[0].centroid)
		    else:
		    	    smallest_distance = getDistance(p,clusters[0].centroid)

		    index = 0
		    for i in range(len(clusters[1:])):
			distance = 0.0
			if dist == 1:
				distance = getDistanceBinary(p,clusters[i+1].centroid)
			else:
			        distance = getDistance(p, clusters[i+1].centroid)
		        if distance < smallest_distance:
		            smallest_distance = distance
		            index = i+1
		    # Add this Point to that Cluster's corresponding list
		    lists[index].append(p)
		# Update each Cluster with the corresponding list
		# Record the biggest centroid shift for any Cluster
		biggest_shift = 0.0
		for i in range(len(clusters)):
		    shift = clusters[i].update(lists[i],dist)
		    biggest_shift = max(biggest_shift, shift)
		# If the biggest centroid shift is less than the cutoff, stop
		if biggest_shift < cutoff: break
    else:
	while True: 
		# Make a list for each Cluster
	  	iterCount =  iterCount + 1
		lists = []
		k=0
		for c in clusters: 
			lists.append([])
			#lists.append([initial[k]])
			#k = k + 1
		# For each Point:
		for p in points:
		    # Figure out which Cluster's centroid is the nearest
		    smallest_distance = 0.0
		    if dist == 1:
			    smallest_distance = getDistanceBinary(p, clusters[0].centroid)
		    else:
		    	    smallest_distance = getDistance(p,clusters[0].centroid)

		    index = 0
		    for i in range(len(clusters[1:])):
			distance = 0.0
			if dist == 1:
				distance = getDistanceBinary(p,clusters[i+1].centroid)
			else:
			        distance = getDistance(p, clusters[i+1].centroid)
		        if distance < smallest_distance:
		            smallest_distance = distance
		            index = i+1
		    # Add this Point to that Cluster's corresponding list
		    lists[index].append(p)
		# Update each Cluster with the corresponding list
		# Record the biggest centroid shift for any Cluster
		biggest_shift = 0.0
		for i in range(len(clusters)):
		    shift = clusters[i].update(lists[i],dist)
		    biggest_shift = max(biggest_shift, shift)
		# If the biggest centroid shift is less than the cutoff, stop
		if biggest_shift < cutoff: break
    # Return the list of Clusters
    return clusters


def getSquaredDistance(a,b):
 # Forbid measurements between Points in different spaces
    if a.n != b.n: raise Exception("ILLEGAL: NON-COMPARABLE POINTS")

    ret = 0.0
    for i in range(a.n):
        ret = ret+pow((a.coords[i]-b.coords[i]), 2)
    return ret

# -- Get distance for binary features
def getSquaredDistanceBinary(a,b):
    
    if a.n != b.n: raise Exception("ILLEGAL: NON COMPARABLE POINTS")
    ret = 0.0
    for i in range(a.n):
    	if a.coords[i] == b.coords[i]:
		ret = ret + 1.0
    
    #print "ret:" , ret		
    ret = ret/a.n
    ret = 1.0 - ret	    
    return ret*ret


# -- Get the Euclidean distance between two Points
def getDistance(a, b):
    # Forbid measurements between Points in different spaces
    if a.n != b.n: raise Exception("ILLEGAL: NON-COMPARABLE POINTS")
    # Euclidean distance between a and b is sqrt(sum((a[i]-b[i])^2) for all i)
    ret = 0.0
    for i in range(a.n):
        ret = ret+pow((a.coords[i]-b.coords[i]), 2)
    return math.sqrt(ret)


# -- Get distance for binary features
def getDistanceBinary(a,b):
    
    if a.n != b.n: raise Exception("ILLEGAL: NON COMPARABLE POINTS")
    ret = 0.0
    for i in range(a.n):
    	if a.coords[i] == b.coords[i]:
		ret = ret + 1.0
    
    #print "ret:" , ret		
    ret = ret/a.n
    ret = 1.0 - ret	    
    return ret

# -- Create a random Point in n-dimensional space
def makeRandomPoint(n, lower, upper):
    coords = []
    for i in range(n): coords.append(random.uniform(lower, upper))
    return Point(coords)

def average(squared_diff_vector):
	return float(sum(squared_diff_vector)/len(squared_diff_vector))

def variance(squared_diff_vector):
	mean = float(sum(squared_diff_vector)/len(squared_diff_vector))
	variance = 0.0
	for x in squared_diff_vector:
		variance = variance + pow((x-mean),2)
	return variance
	

# -- Main function
def main(args):

    kmeans_prepare_data(args[1])
    	

    if len(args) == 1:
    	print "Need file path"
	sys.exit(-1)

    path = "business.csv"	
    option = None
    nclusters = None
    seeds = None

    try:
    	for i in range(len(args)):
    		if args[i] == "-n":
			seeds = int(args[i+1])
		if args[i] == "-k":
			nclusters = int(args[i+1])
		if args[i] == "-b":
			option = "-b"
    except:
    	print "Error in command line options. Please give the options correctly."
	sys.exit(-1)

    #print "Option: ", option
    #print "nclusters: ", nclusters
    #print "seeds:", seeds
    #print "path:", path

    
    csvfile = open(path,'rb')
    reader = csv.reader(csvfile,delimiter='\t')


    num_clusters = [1,2,4,8,15,25]
    points = []

    
    num_points, cutoff = 250, 0.001
    # Create num points random Points in n-dimensional space
    if option == None:
		for row in reader:
			cur_point = []
			colind = 0
			for column in row:
				if colind < 4:
					cur_point.append(float(str(column)))
				colind = colind + 1			
			points.append(Point(cur_point))
    elif option == "-b":
		for row in reader:
			cur_point = []
			colind = 0
			for column in row:
				if colind > 3:
					cur_point.append(float(str(column)))
				colind = colind + 1
			#print "In -b:" , cur_point
			points.append(Point(cur_point))
 
    if seeds == None:	
	    if nclusters == None:
		    for k in num_clusters:
			    clusters = None	
			    if option == "-b":
				    clusters = kmeans(points,k,cutoff,1)
			    else:
			    	    clusters = kmeans(points,k,cutoff,0)

			    squared_diff = 0.0
			    for c in clusters:
			    	for point in c.points:
					if option == "-b":
						squared_diff = squared_diff + getSquaredDistanceBinary(point,c.centroid)
					else:
						squared_diff = squared_diff + getSquaredDistance(point,c.centroid)
					
			    print "Squared Diff: ", squared_diff, "K: ",  k 

	    else:
		clusters = None
		if option == "-b":
			clusters = kmeans(points,nclusters,cutoff,1)
		else:
			clusters = kmeans(points,nclusters,cutoff,0)
		squared_diff = 0.0
		for c in clusters:
			for point in c.points:
				if option == "-b":
						#print "Point: ", point, "Centroid: " , c.centroid , "Distance: ", getDistanceBinary(point,c.centroid)
						squared_diff = squared_diff + getSquaredDistanceBinary(point,c.centroid)
				else:
						squared_diff = squared_diff + getSquaredDistance(point,c.centroid)		
		print "Squared Diff: ", squared_diff, "nclusters: ", nclusters
		if  option == "-b":
			p = get_business(points,nclusters,cutoff,1)
			for x in p:
				print_business(args[1],x,1)
			calculateNMI(points,nclusters,cutoff,1)
		else:
			p = get_business(points,nclusters,cutoff,0)
			for x in p:
				print_business(args[1],x,0)
			calculateNMI(points,nclusters,cutoff,0)


    else:
	    if nclusters == None:
		for k in num_clusters:
			squared_diff_vector = []		
			nclus = 0
    		        for i in range(seeds):
			    	    nclus = k	
				    clusters = None	
			    	    if option == "-b":
				    	clusters = kmeans(points,k,cutoff,1)
			  	    else:
			    	  	  clusters = kmeans(points,k,cutoff,0)
				    squared_diff = 0.0
				    for c in clusters:
				    	for point in c.points:
						if option == "-b":
							squared_diff = squared_diff + getSquaredDistanceBinary(point,c.centroid)					
						else:
							squared_diff = squared_diff + getSquaredDistance(point,c.centroid)						
				    squared_diff_vector.append(squared_diff)		       				  
				    print "Squared Diff: ", squared_diff, "K: ", nclus
			print "Average: " , average(squared_diff_vector), "K: ", nclus
		 	print "Variance: ", variance(squared_diff_vector), "K: ", nclus	 	  

	    else:
			clusters = None			
			squared_diff_vector = []
			for i in range(seeds):
				if option == "-b":
					clusters = kmeans(points,nclusters,cutoff,1)
				else:
					clusters = kmeans(points,nclusters,cutoff,0)
				squared_diff = 0.0
				for c in clusters:
					for point in c.points:
						if option == "-b":
							squared_diff = squared_diff + getSquaredDistanceBinary(point,c.centroid)
						else:
							squared_diff = squared_diff + getSquaredDistance(point,c.centroid)		
				print "Squared Diff: ", squared_diff, "K: ", nclusters
				squared_diff_vector.append(squared_diff)
				if  option == "-b":
					p = get_business(points,nclusters,cutoff,1)
					for x in p:
						print_business(args[1],x,1)
					calculateNMI(points,nclusters,cutoff,1)

				else:
					p = get_business(points,nclusters,cutoff,0)
					for x in p:
						print_business(args[1],x,0)
					calculateNMI(points,nclusters,cutoff,0)
			print "Average: " , average(squared_diff_vector), "K: ", nclusters
		 	print "Variance: ", variance(squared_diff_vector), "K: ", nclusters	  

    
   
 #   for c in clusters: print "C:", c
# -- The following code executes upon command-line invocation
if __name__ == "__main__": main(sys.argv)
