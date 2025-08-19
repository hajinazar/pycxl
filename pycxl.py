#!/usr/bin/env python3

# =========================================================== #
#                           pycxl                             #
#                                                             #
#            Python Script to Calculate Convex Hull           #
#                   and distance above hull                   #
#             for a system of arbitrary dimensions            #
#                                                             #
#  Copyright (c) 2023, Samad Hajinazar                        #
#  samadh~at~buffalo.edu                 v1.8.2 - 08/19/2025  #
# =========================================================== #

#
# Input:
#   1) A text file with "compositions and energies",
#   2) Reference structures don't need to be specified,
#   3) At least one of each elemental entries must be given.
#
# Output(s):
#   1) out_distances.txt: distance above hull for all data,
#   2) out_plot_*.txt:    plot data for binary/ternary systems,
#   3) out_distances.pdf: plot file for binary/ternary systems.
# 
# Example input (binary system):
#   0.0   1.0   0.0
#   1.0   0.0   0.1
#   0.2   0.8  -0.4
#   0.8   0.2   0.9
#
# Example input (ternary system):
#   1.0   0.0   0.0   0.0
#   0.2   0.5   0.3   1.5
#   0.1   0.7   0.2  -0.8
#   0.0   1.0   0.0  -0.1
#   0.0   0.0   1.0   1.3 
#
# A nice compilation of matplotlib "Named Colors":
#   https://stackoverflow.com/a/37232760

import sys
import os.path
import argparse

# ====================================================
# Global variables
# ====================================================
### Internal variables
# Main plot object
plot = None
# Threshold of zero distance above hull: [energy unit]*10^-6
thrs = 1e-6
# Numerical zero (values smaller than this are zero)
zero = 1e-6
# Default value of distance above hull (if failed for a point)
maxh = 99999999999999.0
# The color map for plots (distance and form_ene outputs)
cdst = 'jet'      # jet, viridis, inferno, Reds
cfrm = 'coolwarm' # BrBG
# Various font sizes (not currently in use)
fs_cb_title = 10
fs_cb_label = 8
fs_ax_label = None
fs_tk_label = None

### Command-line adjustable variables
# Resolution of output image (binary and ternary)
idpi = 200
# Make graphic "png" output image type (False for "pdf")
ipng = False
# Remove background of output plot (True for transparent)
ibkg = False
# Edge color for the hull points and tie lines: a valid color name
iclr = "black"
# Input file name
iifl = "points.txt"
# Plot only points on the hull
iohl = False
# Use "formation energy" for symbol colors in plot
ifrm = False
# Create a plain ternary hull plot (no grid mesh)
ipln = False
# Create a plot with no tie lines at all (only points)
inln = False
# Add tags to data points ("inp#" if not given in the input)
itag = False
# Print some debug outputs
idbg = False
# Y-axis range for binary plots (set one/both to valid floats to work)
iyax = [ None, None ]
# Max for color code (set to a valid float to work)
icbr = None
# Plot hull points as square
isqr = False
# No format for hull point shape (disables '-s', '-e' for points)
infm = False
# Fill "all" points with the given uniform color (must have a valid color name)
iunf = None
# Fill hull points with actual color (according to colorbar)
iact = False

# ====================================================
# Check if string variable is a legit number
# ====================================================
def isit_numb_vari(invar):
  try:
    float(invar)
    return True
  except ValueError:
    return False

# ====================================================
# Read points from input file
# ====================================================
def read_inpt_file(flname):
  infile = open(flname, "r")
  inlins = infile.readlines()
  infile.close()

  ### Preliminary check to find dimensionality of the system
  nelem = []
  allns = []
  for line in inlins:
    line = line.strip()
    if line:
      allns.append(line)
      s = line.split()
      if isit_numb_vari(s[0]):
        coun = 0
        while coun < len(s) and isit_numb_vari(s[coun]):
          coun += 1
        nelem.append(coun)

  ### Set the dimensionality
  if len(set(nelem)) != 1:
    print("Error: number of numerical entries is not the same")
    exit()
  ndims = max(nelem)

  ### Read all entries and return the dataset
  all_points = []
  all_idtags = []
  all_labels = []

  sttag = 0
  for line in allns:
    s = line.split()
    if isit_numb_vari(s[0]):
      for j in range(0, ndims):
        all_points.append(float(s[j]))
      # Handle optional identifiers
      idtag = ""
      if itag:
        idtag = "inp"+str(sttag)
        sttag += 1
      if len(s) > ndims:
        idtag = line
        for j in range(0, ndims):
          idtag = idtag.split(maxsplit=1)[1]
        if idtag.startswith("#"):
          idtag = idtag[1:]
      all_idtags.append(idtag.strip())
    elif not s[0].startswith("#"):
      all_labels = s

  all_points = np.asarray(all_points)
  all_points = np.reshape(all_points, (-1, ndims))

  ### Normalize input "coordinates" and "energy" of the points
  for i in range(0, len(all_points)):
    ele_sum = 0.0
    for j in range(0, ndims - 1):
      ele_sum += all_points[i][j]
    for j in range(0, ndims):
      all_points[i][j] /= ele_sum

  return all_points, all_labels, all_idtags

# ====================================================
# Check the input data for some immediate issues
# ====================================================
def chck_inpt_data(inpoints):
  if not isinstance(inpoints, np.ndarray):
    print("Error: input is not nparray")
    exit()
  indim = len(inpoints[0])
  innum = len(inpoints)
  for i in range(0, innum):
    if len(inpoints[i]) != indim:
      print("Error: number of elements is not the same %d" % (i+1))
      print(inpoints[i])
      exit()
    tot_comp = 0.0
    for j in range(0, indim - 1):
      if inpoints[i][j] < 0.0:
        print("Error: fractional composition is negative %d" % (i+1))
        print(inpoints[i])
        exit()
      tot_comp += abs(inpoints[i][j])
    if abs(1.0 - tot_comp) > zero:
      print("Error: sum of fractional compositions is not one %d" % (i+1))
      print(inpoints[i])
      exit()

# ====================================================
# Find elemental reference energies in the dataset
# ====================================================
def find_refs_ener(inpoints):
  dim = len(inpoints[0])

  ### Find the reference elemental energies
  fnd = np.full(dim - 1, False)
  ref = np.full(dim - 1, 0.0)
  lns = np.full(dim - 1, 0.0)
  for i in range(0, len(inpoints)):
    for j in range(0, dim - 1):
      if (1.0 - inpoints[i][j]) < zero:
        if not fnd[j]:
          ref[j] = inpoints[i][dim - 1]
          lns[j] = i
          fnd[j] = True
        if inpoints[i][dim - 1] < ref[j]:
          ref[j] = inpoints[i][dim - 1]
          lns[j] = i

  ### Check if all elemental reference energies are found
  for i in range(0, dim - 1):
    if not fnd[i]:
      print("Error: didn't find ref energy for element %d " % (i+1))
  if not np.all(fnd):
    exit()

  return ref

# ====================================================
# Return adjusted formation energy for a data point
# ====================================================
def calc_form_ener(ref, inpoint):
  frmene = inpoint[-1]
  for i in range(0, len(inpoint) - 1):
    frmene -= inpoint[i] * ref[i]

  return frmene

# ====================================================
# Find the convex hull (after "reducing" the data)
# ====================================================
def find_cnvx_hull(inpoints):
  ### "Scipy Convexhull" needs reduced input:
  #        composition entries without the first one.
  #    But, we do this only here; for the rest full input data is used
  inp_hull = np.delete(np.array(inpoints, copy=True), 0, 1)
  cnv_optn = ""
  return ConvexHull(inp_hull, qhull_options=cnv_optn)

# ====================================================
# Check if a specific point is inside a plane segment
# ====================================================
def isin_hull_segm(inpoints, inpoint):

  n = len(inpoints)
  c = np.zeros(n)
  A = np.r_[inpoints.T, np.ones((1, n))]
  b = np.r_[inpoint,    np.ones(1)]
  l = linprog(c, A_eq = A, b_eq = b)

  return l.success

# ====================================================
# Find distance above hull for a data point
# ====================================================
def find_dist_hull(inhull, inpoint):
  ### Initialize variables
  # A reduced copy of the input (test) point
  newpnt   = np.delete(np.array(inpoint, copy=True), 0)
  # All points
  hll_pnts = inhull.points
  # Dimension of data points (fractional coordinates + energy - 1)
  hll_ndim = len(hll_pnts[0])
  # Number of planes (max actual "length" index: hll_ndim - 1)
  hll_npln = len(inhull.simplices)
  # Number of points per plane (binary: two points, ternary: three points, etc)
  hll_nppp = len(inhull.simplices[0])
  # Hull hyperplane (normal vectors + offsets)
  hll_eqns = inhull.equations

  ### A basic sanity check!
  if len(inhull.simplices) != len(inhull.equations):
    print("Error: issue in find_dist_hull from calculated hull")
    exit()
  if (hll_ndim + 1) != len(inhull.equations[0]):
    print("Error: issue in find_dist_hull from dimensions")
    exit()

  ### Initialize global arrays
  # Distance from the first facets below the point
  above_hull = maxh
  # Intersections of all points with facets
  intersects = np.full((hll_npln, hll_ndim), 0.0)
  # Whether a point intersect facets (and their extension) or not
  doesinters = np.full(hll_npln, True)

  ### Find intersection of all points with all -extended- facets
  for j in range(0, hll_npln):
    last_coor = 0.0
    if abs(inhull.equations[j][hll_ndim - 1]) < zero:
      last_coor = maxh
      doesinters[j] = False
    else:
      for k in range(0, hll_ndim - 1):
        last_coor -= newpnt[k] * inhull.equations[j][k]
      last_coor -= inhull.equations[j][hll_ndim]
      last_coor /= inhull.equations[j][hll_ndim - 1]
      doesinters[j] = True
    # Intersection coordinate corresponding to each point
    for k in range(0, hll_ndim - 1):
      intersects[j][k] = newpnt[k]
    intersects[j][hll_ndim - 1] = last_coor

  ### Check if a point actually intersects a facet (i.e., segment of hyperplane)
  tmppln = np.full((hll_nppp, hll_ndim), 0.0)
  for j in range(0, hll_npln):
    if doesinters[j]:
      # Vertices of the facet (plane segment)
      for m in range(0, hll_nppp):
        for n in range(0, hll_ndim):
          tmppln[m][n] = hll_pnts[inhull.simplices[j][m]][n]
      # Intersection point (hull point and the "extended facet")
      tmppnt = np.array(intersects[j], copy=True)
      # Check if intersection point is within the "segment"
      #   given by set of points "tmppln"
      doesinters[j] = isin_hull_segm(tmppln, tmppnt)

  ### Now, process everything: find "above hull" which is distance from the 
  #     "closest" facet (plane segment) that the point is directly above it.
  #   In general, positive (negative) distance means above (below) hull.
  tmparr = []
  for j in range(0, hll_npln):
    if doesinters[j]:
      tmparr.append(intersects[j][hll_ndim - 1])
  tmparr.sort()
  above_hull = newpnt[hll_ndim - 1] - tmparr[0]
  # A tiny adjustment!
  if abs(above_hull) < thrs:
    above_hull = 0.0

  return above_hull

# ====================================================
# Save the "distance above hull" data to file
# ====================================================
def save_data_dist(inpoints, inenes, indist, intags, flname):
  out = "#"
  for i in range(0, len(inpoints[0]) - 1):
    out += (" % 11s%d" % ("elem", i+1))
  out += ("% 16s" % "orig_ene")
  out += ("% 16s" % "form_ene")
  out += ("% 16s" % "distance")
  out += ("\n")
  for i in range(0, len(inpoints)):
    for j in range(0, len(inpoints[0]) - 1):
      out += (" % 12.6lf" % inpoints[i][j])
    out += ("  % 14.6lf" % inenes[i])
    out += ("  % 14.6lf  % 14.6lf" % (inpoints[i][len(inpoints[0]) - 1], indist[i]))
    out += ("  % s\n" % intags[i]);
  f = open(flname+"_distances.txt", "w")
  f.write(out)
  f.close()

# ====================================================
# Save plot data files for binary and ternary systems
# ====================================================
def save_data_plot(inpoints, indistan, intags, inplanex, inplaney, flname):

  ### Save data points: those on the hull and the rest
  f1=open(flname+"_plot_points.txt", "w")
  f2=open(flname+"_plot_hull_points.txt", "w")

  f1.write("#% 14s   % 14s   % 14s   % 14s\n" % ("x", "y", "form_ene", "distance"))
  f2.write("#% 14s   % 14s   % 14s   % 14s\n" % ("x", "y", "form_ene", "distance"))

  for i in range(0, len(inpoints)):
    if indistan[i] >= thrs:
      f1.write(" % 14.6lf   % 14.6lf   % 14.6lf   % 14.6lf   % s\n" %
           (inpoints[i][0], inpoints[i][1], inpoints[i][2], indistan[i], intags[i]))
    else:
      f2.write(" % 14.6lf   % 14.6lf   % 14.6lf   % 14.6lf   % s\n" %
           (inpoints[i][0], inpoints[i][1], inpoints[i][2], indistan[i], intags[i]))

  f1.close()
  f2.close()

  ### Save hyperplanes' vertices
  f1=open(flname+"_plot_lines.txt", "w") 
  f1.write("#% 14s   % 14s\n" % ("x", "y"))
  for i in range(0, len(inplanex)):
    for j in range(0, len(inplanex[i])):
      f1.write(" % 14.6lf   % 14.6lf\n" % (inplanex[i][j], inplaney[i][j]))
    f1.write("\n")
  f1.close()

# ====================================================
# Return adjusted coords for points and hull vertices
# ====================================================
def calc_data_plot(inhull, indist):
  # Here, we produce adjusted x-y coords of data points and planes, ready
  #   to be plotted in a 2D plot, for both binary and ternary systems.
  # For each data point, we store adjusted coordinates,
  #   formation energy, and distance above hull: [x, y, fene, dist]

  ### Main variables
  # Number of data points
  npnts = len(inhull.points)
  # Dimension of the "hull points" (i.e., actual dims - 1)
  ndims = len(inhull.points[0])
  # Adjusted coordinates of all points
  apnts = np.full((npnts, 4), 0.0)
  # Adjusted coordinates of non-hull points
  opnts = np.empty(shape=[0, 4])
  # Indices of non-hull points
  lopnt = []
  # Adjusted coordinates of points on the hull
  hpnts = np.empty(shape=[0, 4])
  # Indices of points on hull
  lhpnt = []
  # Hyperplane vertices: adjusted x-coords
  hplnx = []
  # Hyperplane vertices: adjusted y-coords
  hplny = []

  ### Find "adjusted" coordinates of all data points (plus their fene/dist)
  for i in range(0, npnts):
    if ndims == 2:
      apnts[i][0] = inhull.points[i][0]
      apnts[i][1] = inhull.points[i][1]
    else:
      apnts[i][0] = (0.5) * (2 * inhull.points[i][0] + inhull.points[i][1])
      apnts[i][1] = (0.5) * (inhull.points[i][1]) * np.sqrt(3.0)
    apnts[i][2] = inhull.points[i][-1] # form ene
    apnts[i][3] = indist[i]            # dist above hull

  ### Make lists of the indices and coordinates+fene+dist for hull/non-hull data points
  for i in range(0, npnts):
    if indist[i] < thrs:
      lhpnt.append(i)
      hpnts = np.vstack((hpnts, apnts[i]))
    else:
      lopnt.append(i)
      opnts = np.vstack((opnts, apnts[i]))
  hpnts = hpnts[hpnts[:, 0].argsort()]
  opnts = opnts[opnts[:, 0].argsort()]

  if idbg: print("\n=== Final hull points\n"); print(lhpnt)
  if idbg: print("\n=== Final hyperplanes\n");

  ### Find hyperplanes (as set of points with adjusted coordinates)
  #   Basically, we find all facets with vertices on the hull
  for i in range(0, len(inhull.simplices)):
    c = 0
    # Check how many vertices are on the hull
    for j in range(0, len(inhull.simplices[i])):
      if inhull.simplices[i][j] in lhpnt:
        c += 1
    # If all vertices are on the hull, add it
    if c == len(inhull.simplices[i]):
      if idbg: print(inhull.simplices[i])
      for j in range(0, len(inhull.simplices[i])):
        hplnx += [apnts[inhull.simplices[i][j]][0]]
        hplny += [apnts[inhull.simplices[i][j]][1]]
      # This needs to be added to get proper ternary plot
      if ndims == 3:
        hplnx += [apnts[inhull.simplices[i][0]][0]]
        hplny += [apnts[inhull.simplices[i][0]][1]]

  # Reformat the hyperplane vertices for proper plot output
  if ndims == 2:
    hplnx = np.reshape(hplnx, (-1, 2))
    hplny = np.reshape(hplny, (-1, 2))
  if ndims == 3:
    hplnx = np.reshape(hplnx, (-1, 4))
    hplny = np.reshape(hplny, (-1, 4))
  if idbg: print("\n=== Hyperplane x coord\n"); print(hplnx)
  if idbg: print("\n=== Hyperplane y coord\n"); print(hplny)

  return apnts, opnts, hpnts, hplnx, hplny

# ====================================================
# Binary-specific plot adjustments
# ====================================================
def hull_plot_binr(plot, inlbls):
  ### Binary specific settings
  #plot.figure(figsize=(5.5,4.0))
  plot.tick_params(axis='both', labelsize=fs_tk_label)
  plot.xlim(0, 1)

  ### Set the end points
  A = (0, 0)
  B = (1, 0)

  ### Plot the lines connecting end points (just to have a dashed line!)
  if (not inln) and (not ipln):
    plot.plot([A[0], B[0]], [A[1], B[1]], linestyle = 'dashed', c=iclr, lw=1)

  ### Generic element names (if they are not give!)
  if len(inlbls) < 2:
    inlbls.append('A')
    inlbls.append('B')

  ### Print labels etc
  plot.ylabel("formation energy", fontsize=fs_ax_label)
  plot.xlabel(r"$\mathrm{x\ in}\ \mathbf{%s}_\mathrm{1-x}\,\mathbf{%s}_\mathrm{x}$" %
              (inlbls[0],inlbls[1]), fontsize=fs_ax_label)

# ====================================================
# Helper function: add ticks for ternary plots
# ====================================================
def make_tick_tern(ax, x1, y1, x2, y2, n_ticks, angle_deg, direction):
  tick_length = 0.02
  label_type  = 2 # (0 none, 1 all, 2 even, 3 odd ticks)
  label_offset= 0.025

  for i in range(1, n_ticks + 1):
    t = i / n_ticks
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    # Set the angles
    theta = np.radians(angle_deg)
    nx = np.cos(theta)
    ny = np.sin(theta)

    # Flip direction if 'inward'
    if direction == 'inward':
      nx *= -1
      ny *= -1

    # Add ticks
    x_start = x
    y_start = y
    x_end   = x + nx * tick_length
    y_end   = y + ny * tick_length
    ax.plot([x_start, x_end], [y_start, y_end], color='black', lw=1, clip_on=False)

    # Label every other one!
    if label_type > 0:
      t_e = i % 2
      if i > 0 and i < n_ticks:
        if label_type == 1 or (label_type == 2 and t_e == 0) or (label_type == 3 and t_e == 1):
          t_n = i / n_ticks
          t_v = f"{t_n:f}"
          t_o = t_v.rstrip('0')
          label_x = x + nx * (tick_length + label_offset)
          label_y = y + ny * (tick_length + label_offset)
          ax.text(label_x, label_y, t_o, fontsize=fs_tk_label, ha='center', va='center', clip_on=False)

# ====================================================
# Ternary-specific plot adjustments
# ====================================================
def hull_plot_tern(plot, inlbls):
  ### Ternary specific settings
  #plot.figure(figsize=(5,4.90))
  plot.axis('off')

  ### Set the end points
  A = (0  , 0       )
  B = (1  , 0       )
  C = (0.5, 0.866025)

  ### Plot the lines connecting end points
  plot.plot([A[0], B[0]], [A[1], B[1]], color=iclr, linewidth=1.0)
  plot.plot([B[0], C[0]], [B[1], C[1]], color=iclr, linewidth=1.0)
  plot.plot([C[0], A[0]], [C[1], A[1]], color=iclr, linewidth=1.0)

  ### Print the label of the end points
  if len(inlbls) < 3:
    inlbls.append('A')
    inlbls.append('B')
    inlbls.append('C')

  plot.text(A[0] - 0.07, A[1] + 0.04, inlbls[0], fontsize=fs_ax_label, weight='bold')
  plot.text(B[0] - 0.05, B[1] - 0.07, inlbls[1], fontsize=fs_ax_label, weight='bold')
  plot.text(C[0] + 0.05, C[1] - 0.01, inlbls[2], fontsize=fs_ax_label, weight='bold')


  ### Add ticks to connecting lines
  nticks = 10

  make_tick_tern(plot, *A, *B, n_ticks=nticks, angle_deg=60 , direction='inward')
  make_tick_tern(plot, *B, *C, n_ticks=nticks, angle_deg=180, direction='inward')
  make_tick_tern(plot, *C, *A, n_ticks=nticks, angle_deg=120, direction='outward')

  ### Plot dashed grid lines if not plain (ternaries only) or no tie lines (all)
  if (not ipln) and (not inln):
    epx = []
    epy = []
    for i in range(1, nticks):
      epx += [ A[0] + i * B[0]/nticks,  C[0] + i * C[0]/nticks,
               B[0] - i * C[0]/nticks,  A[0] + i * C[0]/nticks,
               A[0] + i * C[0]/nticks,  A[0] + i * B[0]/nticks ]
      epy += [ A[1] + i * B[1]/nticks,  C[1] - i * C[1]/nticks ,
               A[1] + i * C[1]/nticks,  A[1] + i * C[1]/nticks ,
               A[1] + i * C[1]/nticks,  A[1] + i * B[1]/nticks ]
    plot.plot(epx, epy, linestyle='dashed', c='darkgray', lw=0.5, zorder=0)

# ====================================================
# Plot convex hull (only binary and ternary systems) 
# ====================================================
def hull_plot_main(inhull, indist, inlabl, intags, flname):
  # Here, we save the "plot-ready" data, only for binary and ternary
  #   systems. Also, if we find matplotlib, we create those plots

  ### Initiate main variables
  numndims = len(inhull.points[0])

  ### This works only for binary and ternary systems
  if numndims != 2 and numndims != 3:
    return

  ### Find adjusted coords for data points and hull plane vertices
  alpoints, nlpoints, hlpoints, hlplanex, hlplaney = calc_data_plot(inhull, indist)

  ### Save output files for points coordinates and lines (e.g., gnuplot)
  save_data_plot(alpoints, indist, intags, hlplanex, hlplaney, flname)

  ### Proceed with producing plot files only if matplotlib was found
  if plot == None:
    return

  ### Collect and process required data (x-y ranges and color-coding info)
  # Extract the default ranges from the actual hull data
  minr = np.sign(alpoints[:,1].min()) * abs(alpoints[:,1].min()) * 1.1
  maxr = np.sign(alpoints[:,1].max()) * abs(alpoints[:,1].max()) * 1.1

  # For x-y ranges, we need adjustments for the binary plots
  if numndims == 2:
    if iyax[0] != None and iyax[1] == None:
      minr = np.sign(alpoints[:,1].min()) * abs(alpoints[:,1].min()) * (1+iyax[0])
      maxr = np.sign(alpoints[:,1].max()) * abs(alpoints[:,1].max()) * (1+iyax[0])
    if iyax[0] != None and iyax[1] != None:
      if iyax[0] > alpoints[:,1].min():
        print("*** Warning: user y min %lf is larger  than data min %lf; using default min" %
              (iyax[0], alpoints[:,1].min()))
      else:
        minr = iyax[0]
      if iyax[1] < alpoints[:,1].min():
        print("*** Warning: user y max %lf is smaller than data min %lf; using default max" %
              (iyax[1], alpoints[:,1].min()))
      else:
        maxr = iyax[1]

  # Prepare the color coding parameters; so we won't have to distinguish
  #  between "above hull" and "formation energy" coloring modes later.
  # Also, for "formation enegy" coloring; we make sure that: min < 0.0 < max
  if ifrm:
    vmin = alpoints[:, -2].min()
    vmax = alpoints[:, -2].max()
    if icbr != None and icbr > vmin:
      vmax = icbr
    #
    if vmin == 0.0:
      vmin -= zero
    if vmax == 0.0:
      vmax += zero
    #
    cb_title = "formation energy"
    cb_norm  = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cb_cmap  = cfrm
    cb_pnts  = ( nlpoints[:, -2], hlpoints[:, -2] )
  else:
    vmin = alpoints[:, -1].min()
    vmax = alpoints[:, -1].max()
    if icbr != None and icbr > vmin:
      vmax = icbr
    cb_title = "distance above hull"
    cb_norm  = colors.Normalize(vmin=vmin, vmax=vmax)
    cb_cmap  = cdst
    cb_pnts  = ( nlpoints[:, -1], hlpoints[:, -1] )

  ### Create the output plot

  # Apply system-specific settings
  if numndims == 2:
    hull_plot_binr(plot, inlabl)
  else:
    hull_plot_tern(plot, inlabl)

  # Plot the tie lines (hull planes)
  if not inln:
    for i in range(0, len(hlplanex)):
      plot.plot(hlplanex[i], hlplaney[i], c=iclr)

  # Plot the points
  mrkr = 'o'  # shape
  size = 70   # size
  ewid = 1    # edge width

  # (1) A "place-holder" to setup the colorbar
  ps= plot.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=0, alpha=0.7,  edgecolors=None, linewidth=ewid,
                   s=0   , marker=mrkr, c=cb_pnts[1], cmap=cb_cmap, norm=cb_norm)

  # (2) The plot for non-hull points
  if not iohl:
    if not infm and iunf:
      plot.scatter(nlpoints[:, 0], np.ma.masked_outside(nlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=None, alpha=0.7,  edgecolors=None, linewidth=ewid,
                   s=size, marker=mrkr, c=iunf      , cmap=None   , norm=None)
    else:
      plot.scatter(nlpoints[:, 0], np.ma.masked_outside(nlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=None, alpha=0.7,  edgecolors=None, linewidth=ewid,
                   s=size, marker=mrkr, c=cb_pnts[0], cmap=cb_cmap, norm=cb_norm)

  # (3) The plot for hull points (with/without formatting options)
  if infm:
    plot.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr),
                 clip_on=False, zorder=None, alpha=0.7,  edgecolors=None, linewidth=ewid,
                 s=size, marker=mrkr, c=cb_pnts[1], cmap=cb_cmap, norm=cb_norm)
  else:
    #
    if not iclr:
      ewid = 0
    if isqr:
      mrkr = 's'
      size = 85
    #
    size = size - ewid
    if iunf:
      plot.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=5, alpha=None, edgecolors=iclr, linewidth=ewid,
                   s=size, marker=mrkr, c=iunf      , cmap=None   , norm=None)
    elif iact:
      plot.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=5, alpha=None, edgecolors=iclr, linewidth=ewid,
                   s=size, marker=mrkr, c=cb_pnts[1], cmap=cb_cmap, norm=cb_norm)
    else:
      plot.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr),
                   clip_on=False, zorder=5, alpha=None, edgecolors=iclr, linewidth=ewid,
                   s=size, marker=mrkr, c='white'   , cmap=None   , norm=None)

  # Add the colorbar (for uniform colors, only a placeholder)
  if iunf:
    cb = plot.colorbar(ps, label="" , boundaries = [-zero, +zero])
    cb.ax.tick_params(axis='y', length=0, labelright=False)
  else:
    cb = plot.colorbar(ps, label=cb_title)
    cb.ax.set_yscale('linear')

  # Apply plot range limits
  plot.ylim(minr, maxr)

  ### Save the plot to the output file
  if ipng:
    plot.savefig(flname+"_distances.png", dpi=idpi, transparent=ibkg)
  else:
    plot.savefig(flname+"_distances.pdf", dpi=idpi, transparent=ibkg)

# ====================================================
# Print the program header
# ====================================================
def prnt_prog_hdrs():
  print("=============================================")
  print("pycxl: Python Script to Calculate Convex Hull")
  print("                                             ")
  print("Samad Hajinazar  samadh~at~buffalo.edu v1.8.2")
  print("=============================================")
  print()

# ====================================================
# Main entry for command line task: process input, ...
# ====================================================
def chck_inpt_args():
  ### Use global variables for possible updating through command-line options
  global iifl, ifrm, ipng, ibkg, idpi, iyax, icbr, iclr, ipln, inln, iohl, isqr, iact, iunf, infm, itag, idbg

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=iifl, help="Input file name")
  parser.add_argument("-f", "--formene", action="store_true", default=ifrm, help="Use formation enes to color points")
  parser.add_argument("-g", "--graphics", action="store_true", default=ipng, help="Output png graphics instead of pdf")
  parser.add_argument("-b", "--background", action="store_true", default=ibkg, help="Transparent background in output plot")
  parser.add_argument("-r", "--resolution", type=int, default=idpi, help="Resolution (dpi) of output plot")
  parser.add_argument(
    "-y", "--yrange",
    nargs="*",                # accept 0,1,2,… floats
    type=float,
    default=None,             # so we can detect “flag omitted” vs. “flag given but no values”
    metavar=("Y0", "Y1"),
    help="y range - '-y 0.3' : relative change in actual values; '-y 1.0 2.5' : explicit values of min/max"
  )
  parser.add_argument("-c", "--colorbar", type=float, default=icbr, help="Max value of colorbar")
  parser.add_argument("-p", "--plain", action="store_true", default=ipln, help="Plot for ternaries with no grid lines")
  parser.add_argument("-l", "--lineless", action="store_true", default=inln, help="Plot with no tie line: only points")
  parser.add_argument("-o", "--onlyhull", action="store_true", default=iohl, help="Plot only hull points")
  parser.add_argument("-e", "--edgecolor", type=str, default=iclr, help="Color hull points and tie lines with the given 'Named' color")
  parser.add_argument("-a", "--actualcolor", action="store_true", default=iact, help="Fill hull points with actual colors")
  parser.add_argument("-u", "--uniformcolor", type=str, default=iunf, help="Fill all points with the given 'Named' color")
  parser.add_argument("-s", "--squarepoints", action="store_true", default=isqr, help="Show hull points as square")
  parser.add_argument("-n", "--noformathull", action="store_true", default=infm, help="No shape format for hull points")
  parser.add_argument("-t", "--tags", action="store_true", default=itag, help="Add tags to all input data")
  parser.add_argument("-d", "--debug", action="store_true", default=idbg, help="Print debug info")
  args = parser.parse_args()

  # Process the yrange for binary plots
  if args.yrange is None:
    pass
  elif len(args.yrange) == 1: # one float: offset
    if args.yrange[0] <= 1.0 and args.yrange[0] >= 0.0:
      iyax[0] = args.yrange[0]
    else:
      parser.error("argument -y/--yrange: the offset must be between 0 and 1")
  elif len(args.yrange) == 2: # two floats: min,max
    if args.yrange[1] > args.yrange[0]:
      iyax = args.yrange
    else:
      parser.error("argument -y/--yrange: max should be larger than min")
  else:
    parser.error("argument -y/--yrange accepts only up to two floats values")

  # Process other inputs
  icbr = args.colorbar
  ipng = args.graphics
  ifrm = args.formene
  iohl = args.onlyhull
  ipln = args.plain
  inln = args.lineless
  ibkg = args.background
  iclr = args.edgecolor
  idpi = args.resolution
  iifl = args.input
  isqr = args.squarepoints
  iact = args.actualcolor
  iunf = args.uniformcolor
  infm = args.noformathull
  itag = args.tags
  idbg = args.debug

# ====================================================
# Main entry for command line task: process input, ...
# ====================================================
def main_cmdl_task():
  ### Read all points from the input file
  inp_data, inp_lbls , inp_tags = read_inpt_file(iifl)

  ### Verify the input data
  chck_inpt_data(inp_data)

  ### Calculate reference elemental energies and formation energies
  inp_enes = np.array(inp_data[:, len(inp_data[0]) - 1], copy=True)
  ref_enes = find_refs_ener(inp_data)
  for i in range(0, len(inp_data)):
    inp_data[i][-1] = calc_form_ener(ref_enes, inp_data[i])

  ### Calculate convex hull
  cnv_hull = find_cnvx_hull(inp_data)
  if idbg: print("\n=== Hull simplices\n"); print(cnv_hull.simplices)
  if idbg: print("\n=== Hull equations\n"); print(cnv_hull.equations)
  if idbg: print("\n=== Hull vertices \n"); print(cnv_hull.vertices)

  ### Calculate the distances of (all) points from the hull
  out_dist = []
  for i in range(0, len(inp_data)):
    out_dist.append(find_dist_hull(cnv_hull, inp_data[i]))
  out_dist = np.array(out_dist)

  ### Save the main results: points and their distance above hull
  save_data_dist(inp_data, inp_enes, out_dist, inp_tags, 'out')

  ### Check if it's a binary or ternary to proceed with plot files
  ndim = len(inp_data[0]) - 1
  if ndim != 2 and ndim != 3:
    return

  ### Save the 2D plot-ready hull data files and -possibly- output plot
  hull_plot_main(cnv_hull, out_dist, inp_lbls, inp_tags, 'out')

# ====================================================
# Command line call
# ====================================================
if __name__ == '__main__':
  ### Intro ...
  prnt_prog_hdrs()
  chck_inpt_args()

  ### Check if the input file exist
  if not os.path.isfile(iifl):
    print("Error: input file '%s' does not exist" % (iifl))
    exit()
  else:
    print("Found input file '%s' ...\n" % (iifl))

  ### Check if required/optional modules exist
  try:
    import numpy as np
  except ImportError:
    print("Error: failed to load 'numpy'")
    exit()
  try:
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    from scipy.optimize import linprog
  except ImportError:
    print("Error: failed to load 'scipy'")
    exit()
  try:
    import matplotlib.pyplot as plot
    from matplotlib import colors
  except ImportError:
    print("*** Warning: failed to load 'matplotlib'- no output plot will be created\n")

  ### Check if given colors are valid (only if matplotlib is found)
  if plot:
    if not colors.is_color_like(iclr):
      print("Error: the color name '%s' for edges  is not valid\n" % iclr)
      exit()
    if iunf != None and (not colors.is_color_like(iunf)):
      print("Error: the color name '%s' for points is not valid\n" % iunf)
      exit()

  ### Main task
  main_cmdl_task()

  print("All done!")
  exit()
