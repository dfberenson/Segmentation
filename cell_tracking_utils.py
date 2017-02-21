# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:26:41 2017

@author: dfberenson@gmail.com

@ToDo: Convert data to numpy arrays rather than lists
@ToDo: Use centroid-based scheme for mapping off-target clicks rather than expanding grid

"""


#Takes filename for Excel file as input and returns contents as a sheetwise list of dataframes
def XlsxReader(filename):
    import pandas
    xlsx = pandas.ExcelFile(filename)         
    data = []
    for i in range(len(xlsx.sheet_names)):
        data.append(xlsx.parse(xlsx.sheet_names[i]))
    return data

#Takes filename for Excel file as input and returns a list of the names of constituent sheets as unicodes
def XlsxSheetNames(filename):
    import pandas
    xlsx = pandas.ExcelFile(filename)    
    sheet_names = xlsx.sheet_names
    return sheet_names

#Takes filename for Excel file as input and returns a dictionary where each sheet name (a cell number starting with 1000)
#is keyed to a list in the form [[frame,[x,y]]] -- only does this for cells that appear in this particular stack
def TrackingDataDictionary(filename, firstcell, lastcell):
    import pandas
    data = XlsxReader(filename)
    sheet_names = map(int, XlsxSheetNames(filename))

    d = {}
    for i in range(len(sheet_names)):
        if firstcell <= sheet_names[i] <= lastcell:
            frames = list(data[i]['Frame'])
            X = list(data[i]['X'])
            Y = list(data[i]['Y'])
                
            track_coordinates = []
                
            for j in range(len(frames)):
                track_coordinates.append([frames[j] , [X[j],Y[j]]])
                
            cellnum = sheet_names[i]
            d[cellnum] = track_coordinates
         
         
    return d


#Takes 'labels' pixel matrix with originally assigned numerical labels for each cell and reassigns the numerical label
#according to the cellnum from the cell data dictionary
def RenameLabels(labels , cell_dict):
    for cell in cell_dict:
        if cell == 'notes':
            print cell_dict[cell]
        else:
            cell_data = cell_dict[cell]
            #cell_data is a list in the format [frame, [x,y]]
            cellnum = cell
            lifespan = len(cell_data)
            first_frame = cell_data[0][0] - 1                                  #finds the right frame index for the first frame in which the cell appears
            for f in range(lifespan):
                xy = cell_data[f][1]                                           #gets the xy coordinates for the current cell
                xy_near = xy
                dist = 0
                t = first_frame + f                                               #adjusts the frame index to match the image stack
                curr_labels = labels[:,:,t]                                       #gets the label matrix for this frame
                cell_labelnum = curr_labels[xy[1],xy[0]]                          #finds the original label number of the tracked cell.
                                           #Note we need to address the coordinates as (Y,X)
                                           
                while cell_labelnum == 0:
                    dist += 1
                    
                    for x_adj in range(-dist,dist+1):
                        xy_near = [xy[0] + x_adj , xy[1] + dist]
                        cell_labelnum = curr_labels[xy_near[1],xy_near[0]]
                        if cell_labelnum != 0:
                            break
                        xy_near = [xy[0] + x_adj , xy[1] - dist]
                        cell_labelnum = curr_labels[xy_near[1],xy_near[0]]
                        if cell_labelnum != 0:
                            break
                        
                    if cell_labelnum == 0:
                       for y_adj in range(-dist,dist+1):
                            xy_near = [xy[0] + dist , xy[1] + y_adj]
                            cell_labelnum = curr_labels[xy_near[1],xy_near[0]]
                            if cell_labelnum != 0:
                                break
                            xy_near = [xy[0] - dist , xy[1] + y_adj]
                            cell_labelnum = curr_labels[xy_near[1],xy_near[0]]
                            if cell_labelnum != 0:
                                break           
   
                cell_labelnum = curr_labels[xy_near[1],xy_near[0]]
                curr_labels[curr_labels == cell_labelnum] = cellnum          #replaces the original label number with the official label number                    
                labels[:,:,t] = curr_labels
    return labels




#Takes filename for Excel file as input and returns a sheetwise list of timepoints and coordinates in the form [frame , [x,y]]
def TrackingDataReader(filename):
    if filename[-4:] == 'xlsx':
        import pandas
        data = XlsxReader(filename)
        sheet_names = XlsxSheetNames(filename)
        all_coordinates = []
        
        for i in range(len(sheet_names)):
            frames = list(data[i]['Frame'])
            X = list(data[i]['X'])
            Y = list(data[i]['Y'])
            
            track_coordinates = []
            
            for j in range(len(frames)):
               track_coordinates.append([frames[j] , [X[j],Y[j]]])              
    
            all_coordinates.append(track_coordinates)
    
        return all_coordinates
    
    
    
    
#    
#    
#    elif filename[-3:] == 'csv':
#        import csv
#        import math
#        
#        csvfile = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\Manual Cell Tracking\Cell 3.csv'
#        
#        with open (csvfile, 'rb') as f:
#            data = list(csv.reader(f))
#        f.close
#        
#        #Get vector of column names, then remove top row
#        headers = data[0]
#        del data[0]
#        
#        #Assign data to appropriately named variables
#        X = []
#        Y = []
#        for i in range(len(headers)):
#            for t in range(len(data)):
#                if headers[i] == 'X':
#                    X.append(int(data[t][i]))
#                if headers[i] == 'Y':
#                    Y.append(int(data[t][i]))
#        
#        #Combine X and Y coordinates into a list of coordinate vectors
#        coorVectors = []
#        for t in range(len(data)):
#            coorVectors.append([X[t] , Y[t]])
#            
#        #Calculate velocities
#        velocities = [[0,0]]
#        for t in range(1 , len(data)):
#            velocities.append([X[t] - X[t-1] , Y[t] - Y[t-1]])
#            
#        #Calculate speeds
#        speeds = [0]
#        for t in range(1 , len(data)):
#            speeds.append(math.sqrt(velocities[t][0]**2 + velocities[t][1]**2))
#            
#        #Calculate angles
#        angles = [0]
#        for t in range(1 , len(data)):
#            angles.append(math.atan2(velocities[t][1] , velocities[t][0]))
#            
#        #Combine speed and angular coordinates into a list of angular velocity vectors as [r , theta]
#        angvelocities = []
#        for t in range(len(data)):
#            angvelocities.append([speeds[t] , angles[t]])
#            
#        return coorVectors