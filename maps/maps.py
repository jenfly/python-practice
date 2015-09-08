import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime

# Globe with Orthographic projection
# ----------------------------------
# lon_0, lat_0 are the center point of the projection.
# resolution = 'l' means use low resolution coastlines.
lon_0, lat_0 = -105, 40
#lon_0, lat_0 = -105, 90
plt.figure()
m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
m.drawmapboundary(fill_color='aqua')
plt.title("Full Disk Orthographic Projection")

# Hammer Projection
# -----------------
# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
plt.figure()
m = Basemap(projection='hammer',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
m.drawmapboundary(fill_color='aqua')
plt.title("Hammer Projection")

# Robinson Projection
# -------------------
# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
plt.figure()
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary(fill_color='aqua')
plt.title("Robinson Projection")


# Shaded Relief Map and Day/Night Shading
# --------------------------------------
lon_0, lat_0 = -60, 0
date = datetime(2014, 12, 22, 9, 55)
#date = datetime(2014, 7, 22, 9, 55)
scale = 0.2
plt.figure()
m = Basemap(projection='ortho', lat_0=lat_0, lon_0=lon_0)
m.shadedrelief(scale=scale)
m.nightshade(date)
plt.title('Shaded Relief with Day/Night')

# North and South Pole
# -----------------------------------------
lon_0 = -50
scale = 0.2
plt.figure(figsize=(7,10))
plt.subplot(211)
m = Basemap(projection='ortho', lat_0=90, lon_0=lon_0)
m.shadedrelief(scale=scale)
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
plt.title('North Pole View')
plt.subplot(212)
m = Basemap(projection='ortho', lat_0=-90, lon_0=lon_0)
m.shadedrelief(scale=scale)
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
plt.title('South Pole View')

# NASA's Blue Marble
# ------------------
plt.figure()
m = Basemap()
m.bluemarble()

# Etopo
# -----
plt.figure()
m = Basemap()
m.etopo()
