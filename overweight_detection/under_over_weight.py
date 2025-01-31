### Finding the relation between the fish and their weight
# the following formula is going to give us the length:  
#   > pixel/cm --> total_pixels/ pixel/cm   
#   > cm/pixel formula --> cm / pixel*total_pixels
def fish_length(width_box, WIDTH_IMG):
    RULER_WIDTH=8470.0/164.0
    return  (width_box*WIDTH_IMG)/RULER_WIDTH

#(weight*100)/length^3 is our relation. 1-1.3 is going to be the healthy range
def percentage(length, weigth):
    return ((weigth)*100)/length**3

# Classify weight
def under_over_weight(width_box : int, WIDTH_IMG : int, weight : int):
    res = percentage(fish_length(width_box, WIDTH_IMG), weight)
    if( 1 <= res or res <= 1.3): 
        # print("HEALTHY FISH")                                                              
        return 1 # 1 for HEALTHY FISH
    if( 1 > res ):
        # print("UNDERWEIGHT FISH")
        return 0 # 0 for UNDERWEIGHT FISH
    else:
        # print("OVERWEIGHT FISH")
        return 2 # 2 for OVERWEIGHT FISH
                
                