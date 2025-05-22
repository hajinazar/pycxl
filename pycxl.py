#!/usr/bin/env python3

# =========================================================== #
#                           pycxl                             #
#                                                             #
#    Python script to calculate distance above convex hull    #
#             for a system of arbitrary dimensions            #
#                                                             #
#  Copyright (c) 2023, Samad Hajinazar                        #
#  samadh~at~buffalo.edu                   v1.4 - 05/22/2025  #
# =========================================================== #

#
# Input:
#   1) A text file with "fractional compositions and energies",
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

import sys
import os.path
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import linprog

# ====================================================
# Global variables
# ====================================================
# Default resolution of output image (binary and ternary)
idpi = 200
# Default output image type: "png" "pdf"
ifig = "pdf"
# Default input file name
ifil = "points.txt"
# Threshold of zero distance above hull: [energy unit]*10^-6
thrs = 1e-6
# Numerical zero (values smaller than this are zero)
zero = 1e-6
# Infinity
infi = float('inf')
# Default value of distance above hull (if failed for a point)
maxh = 99999999999999.0
# Plot only points on the hull
plth = False
# Use "formation energy" for symbol colors in plot
pltf = False
# Create a plain plot (no grid mesh)
pltp = False
# Always add the tag ("inp#" if not given in the input)
ptag = False
# Print some debug outputs
dbug = False
# The color map for plots (distance and form_ene outputs)
cdst = 'jet'      # jet, viridis, inferno, Reds
cfrm = 'coolwarm' # BrBG
# Default ranges: applicable only to binary plots
rang = [ infi, infi ] # y-range (set both to != infi to work)
maxc = infi # max for color code (set to != infi to work)

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
      if ptag:
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
# Save the "distance above hull" data to file
# ====================================================
def save_dist_data(inpoints, inenes, indist, intags, flname):
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
def save_plot_data(inpoints, indistan, intags, inplanex, inplaney, flname):

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
# Binary-specific plot adjustments
# ====================================================
def hull_plot_binr(iplt, inlbls):
  ### Binary specific settings
  if (plth) and (not pltf):
    iplt.figure(figsize=(5.5,4.0))

  ### Generic element names (if they are not give!)
  if len(inlbls) < 2:
    inlbls.append('A')
    inlbls.append('B')

  ### Plot lines connecting end points (just to have a dashed line!)
  iplt.xlim(0, 1)
  epx0 = [ 0.000000, 1.000000 ]
  epy0 = [ 0.000000, 0.000000 ]
  iplt.plot(epx0[0:2], epy0[0:2], linestyle = 'dashed', c='black', lw=1)

  ### Print labels etc
  iplt.ylabel("formation energy")
  iplt.xlabel("x in %s\u2081\u208B\u2093%s\u2093" % (inlbls[0],inlbls[1]))

# ====================================================
# Ternary-specific plot adjustments
# ====================================================
def hull_plot_tern(iplt, inlbls):
  ### Ternary specific settings
  if (plth) and (not pltf):
    iplt.figure(figsize=(5,4.90))
  iplt.axis('off')

  ### Plot lines connecting end points (just to have ticks!)
  epx0 = [ 0.000000, 1.000000, 0.500000, 0.000000 ]
  epy0 = [ 0.000000, 0.000000, 0.866025, 0.000000 ]
  from matplotlib import patheffects
  iplt.plot(epx0[0:2], epy0[0:2], linestyle='solid', c='black', lw=1,
            path_effects=[patheffects.withTickedStroke
            (angle=242, spacing=25, length=.25, offset=(12.5,-1))])
  iplt.plot(epx0[1:3], epy0[1:3], linestyle='solid', c='black', lw=1,
            path_effects=[patheffects.withTickedStroke
            (angle=242, spacing=26, length=.25, offset=(-6,12.5))])
  iplt.plot(epx0[2:4], epy0[2:4], linestyle='solid', c='black', lw=1,
            path_effects=[patheffects.withTickedStroke
            (angle=235, spacing=26, length=.25, offset=(-7,-11.5))])

  ### Print the label of the end points
  if len(inlbls) < 3:
    inlbls.append('A')
    inlbls.append('B')
    inlbls.append('C')

  iplt.text(epx0[0] - 0.07, epy0[0] + 0.04, inlbls[0], weight='bold')
  iplt.text(epx0[1] - 0.05, epy0[1] - 0.07, inlbls[1], weight='bold')
  iplt.text(epx0[2] + 0.05, epy0[2] - 0.01, inlbls[2], weight='bold')

  ### Plot dashed mesh lines if not plain figure (ternaries only)
  if not pltp:
    epx1 = []
    epy1 = []
    for i in range(1, 10):
      epx1 += [ 0.000000 + i * 0.1000000, 0.500000 + i * 0.0500000 ,
                1.000000 - i * 0.0500000, 0.000000 + i * 0.0500000 ,
                0.000000 + i * 0.0500000, 0.000000 + i * 0.1000000 ]
      epy1 += [ 0.000000                , 0.866025 - i * 0.0866025 ,
                0.000000 + i * 0.0866025, 0.000000 + i * 0.0866025 ,
                0.000000 + i * 0.0866025, 0.000000                 ]
    iplt.plot(epx1, epy1, linestyle='dashed', c='darkgray', lw=0.5, zorder=0)

# ====================================================
# Return adjusted coords for points and hull vertices
# ====================================================
def hull_plot_data(inhull, indist):
  ### Main variables
  # Number of data points
  npnts = len(inhull.points)
  # Dimension of the "hull points" (i.e., actual dims - 1)
  ndims = len(inhull.points[0])
  # Adjusted coordinates of all points
  apnts = np.full((npnts, 3), 0.0)
  # Adjusted coordinates of points on the hull
  hpnts = np.empty(shape=[0, 3])
  # Indices of points on hull
  lhpnt = []
  # Hyperplane vertices: adjusted x-coords
  hplnx = []
  # Hyperplane vertices: adjusted y-coords
  hplny = []

  ### Find "adjusted" coordinates of all data points
  for i in range(0, npnts):
    if ndims == 2:
      apnts[i][0] = inhull.points[i][0]
      apnts[i][1] = inhull.points[i][1]
    else:
      apnts[i][0] = (0.5) * (2 * inhull.points[i][0] + inhull.points[i][1])
      apnts[i][1] = (0.5) * (inhull.points[i][1]) * np.sqrt(3.0)
    apnts[i][2] = inhull.points[i][-1]

  ### Find the indices and coordinates of points on the convex hull
  for i in range(0, npnts):
    if indist[i] < thrs:
      lhpnt.append(i)
      hpnts = np.vstack((hpnts, apnts[i]))
  hpnts = hpnts[hpnts[:, 0].argsort()]

  if dbug: print("\n=== Final hull points\n"); print(lhpnt)
  if dbug: print("\n=== Final hyperplanes\n");

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
      if dbug: print(inhull.simplices[i])
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
  if dbug: print("\n=== Hyperplane x coord\n"); print(hplnx)
  if dbug: print("\n=== Hyperplane y coord\n"); print(hplny)

  return apnts, hpnts, hplnx, hplny

# ====================================================
# Plot convex hull (only binary and ternary systems) 
# ====================================================
def hull_plot_main(inhull, indist, inlabl, intags, flname):
  ### Initiate main variables
  numpoint = len(inhull.points)
  numndims = len(inhull.points[0])

  ### Proceed only for binary and ternary systems
  if numndims != 2 and numndims != 3:
    exit()

  ### Find adjusted coords for data points and hull plane vertices
  alpoints, hlpoints, hlplanex, hlplaney = hull_plot_data(inhull, indist)

  ### Save output files for points coordinates and lines (e.g., gnuplot)
  save_plot_data(alpoints, indist, intags, hlplanex, hlplaney, flname)

  ### Proceed creating plots only if matplotlib exists
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    exit()

  # Initiate colorbar info: label + (cval, cmap, cbar) for allpoints and hullpoints
  from matplotlib import colors
  if not pltf:
    vmin = indist.min()
    if maxc != infi and maxc > vmin:
      vmax = maxc
    else:
      vmax = indist.max()
    pvar = ("distance above hull",
            indist, cdst, colors.Normalize(vmin=vmin, vmax=vmax),
            'white', None, None)
  else:
    vmin = alpoints[:, -1].min()
    vmax = alpoints[:, -1].max()
    if maxc != infi and maxc > vmin:
      vmax = maxc
    else:
      vmax = indist.max()
    # This is to satisfy the requirement of vmin < 0.0 < vmax for this type of plots
    if vmin == 0.0:
      vmin -= zero
    if vmax == 0.0:
      vmax += zero
    pvar = ("formation energy",
       alpoints[:, -1], cfrm, colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
       hlpoints[:, -1], cfrm, colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax))

  # Start with system specific parts of the plots
  if numndims == 2:
    hull_plot_binr(plt, inlabl)
  else:
    hull_plot_tern(plt, inlabl)

  # Plot all points
  ps = None

  # Extract the default ranges from the actual hull data
  minr = np.sign(alpoints[:,1].min()) * abs(alpoints[:,1].min()) * 1.1
  maxr = np.sign(alpoints[:,1].max()) * abs(alpoints[:,1].max()) * 1.1

  # Adjustments to plot range (only for 2D hull)

  if numndims == 2 and (rang[0] != infi and rang[1] != infi and rang[1] > rang[0]):
    if rang[0] <= alpoints[:,1].min():
      minr = rang[0]
      maxr = rang[1]
    else:
      print("Warning: actual data limits are: % lf and % lf! Setting only maximum." % (minr, maxr))
      maxr = rang[1]

  if not plth:
    ps = plt.scatter(alpoints[:, 0], np.ma.masked_outside(alpoints[:, 1], minr, maxr), alpha=0.7, s=70,
                     c=pvar[1], cmap=pvar[2], norm=pvar[3], clip_on=False)

  # Plot hull points
  plt.scatter(hlpoints[:, 0], np.ma.masked_outside(hlpoints[:, 1], minr, maxr), s=70, clip_on=False,
              c=pvar[4], edgecolors='black', cmap=pvar[5], norm=pvar[6], zorder=4)

  # Plot hull planes
  for i in range(0, len(hlplanex)):
    plt.plot(hlplanex[i], hlplaney[i], c='black')

  # Final adjustment: colorbar
  if (not plth) or (pltf):
    cb = plt.colorbar(ps, label=pvar[0])
    cb.ax.set_yscale('linear')

  # Apply plot range limits
  plt.ylim(minr, maxr)

  ### Save the plot file
  plt.savefig(flname+"_distances."+ifig, dpi=idpi)

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
  # Number of points per plane (2D: two points, 3D: three points, etc)
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
# Print the program header
# ====================================================
def prnt_prog_hdrs():
  print("=====================================================")
  print("pycxl: Python script to calculate distance above hull")
  print("                                                     ")
  print("Samad Hajinazar      samadh~at~buffalo.edu       v1.4")
  print("=====================================================")
  print()

# ====================================================
# Main entry for command line task: process input, ...
# ====================================================
def main_cmdl_task():
  ### Use global variables for possible input values
  global ifil, ifig, plth, pltf, pltp, ptag, dbug, rang, maxc

  ### Read the input variables
  if len(sys.argv) >= 2:
    cmdl = sys.argv[1:]
    skip_next = False
    for i in range(0, len(cmdl)):
      if skip_next:
        skip_next = False
        continue
      if cmdl[i] == '-f':
        pltf = True
      elif cmdl[i] == '-h':
        plth = True
      elif cmdl[i] == '-p':
        pltp = True
      elif cmdl[i] == '-t':
        ptag = True
      elif cmdl[i] == '-d':
        dbug = True
      elif cmdl[i] == '-g':
        ifig = "png"
      elif cmdl[i] == '-r':
        l = []
        for t in cmdl[i+1].split():
          try:
            l.append(float(t))
          except ValueError:
            pass
        if len(l) == 2:
          rang[0] = l[0]; rang[1] = l[1]
        skip_next = True
      elif cmdl[i] == '-c':
        l = []
        try:
          l.append(float(cmdl[i+1].split()[0]))
        except ValueError:
          pass
        if len(l) == 1:
          maxc = l[0]
        skip_next = True
      elif ifil == "points.txt":
        ifil = cmdl[i]

  ### Check if input file exists
  if not os.path.isfile(ifil):
    print("Error: input file '%s' does not exist" % (ifil))
    exit()

  ### Read all points from the input file
  inp_data, inp_lbls , inp_tags = read_inpt_file(ifil)

  ### Verify the input data
  chck_inpt_data(inp_data)

  ### Calculate reference elemental energies and formation energies
  inp_enes = np.array(inp_data[:, len(inp_data[0]) - 1], copy=True)
  ref_enes = find_refs_ener(inp_data)
  for i in range(0, len(inp_data)):
    inp_data[i][-1] = calc_form_ener(ref_enes, inp_data[i])

  ### Calculate convex hull
  cnv_hull = find_cnvx_hull(inp_data)
  if dbug: print("\n=== Hull simplices\n"); print(cnv_hull.simplices)
  if dbug: print("\n=== Hull equations\n"); print(cnv_hull.equations)
  if dbug: print("\n=== Hull vertices \n"); print(cnv_hull.vertices)

  ### Calculate the distances of (all) points from the hull
  out_dist = []
  for i in range(0, len(inp_data)):
    out_dist.append(find_dist_hull(cnv_hull, inp_data[i]))
  out_dist = np.array(out_dist)

  ### Print the results: points and their distance above hull
  save_dist_data(inp_data, inp_enes, out_dist, inp_tags, 'out')

  ### Save the convex hull plot to file (only binary and ternary)
  hull_plot_main(cnv_hull, out_dist, inp_lbls, inp_tags, 'out')

# ====================================================
# Command line call
# ====================================================
if __name__ == '__main__':
  prnt_prog_hdrs()
  main_cmdl_task()
  print("All done!")
  exit()
