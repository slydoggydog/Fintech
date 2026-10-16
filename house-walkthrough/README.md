# 419 Jersey Avenue: 3D walkthrough

A walkable 3D model of 419 Jersey Ave, Spring Lake, NJ (1953 Cape Cod), rebuilt
from the listing photos and details. You control a person and can walk
through every room, up the stairs, out to the screened porch and the backyard.

## Open it

**From your computer:** download this `house-walkthrough` folder and
double-click `index.html`. It runs in Chrome, Edge, Safari or Firefox with
no install or internet connection needed.

**As a link you can share:** enable GitHub Pages on this repository
(Settings → Pages → Deploy from branch → `main`, folder `/ (root)`) once this
is merged. The walkthrough will then be at
`https://<your-username>.github.io/<repo>/house-walkthrough/`.

## Controls

| Action | Keyboard / mouse | Phone / tablet |
| --- | --- | --- |
| Walk | W A S D or arrow keys | Drag with left thumb |
| Jog | Hold Shift | Push the joystick all the way |
| Look around | Drag with the mouse | Drag with right thumb |
| Zoom camera | Scroll wheel | |
| First / third person | V or the button on the right | Button on the right |

The **Go to…** menu jumps straight to a room. **Change outfit** recolors the
character.

## What's modeled

- Exterior: cedar shake siding, navy shutters, window boxes, striped awning
  over the front porch, bay window, brick chimney, rear shed dormer.
- First floor: foyer and staircase, living room with fireplace, blue dining
  room, kitchen with skylight, screened porch, front bedroom used as a TV
  room, two more bedrooms, hall bath.
- Upstairs: two large bedrooms and a bath.
- Yard: 150' deep lot, flagstone patio, outdoor shower, Bilco basement
  doors, picnic table, shed, gravel lane out the back.

Room sizes and anything not visible in the listing photos (the upstairs
layout especially) are best guesses. Measurements or more photos would make
the model more accurate.

## Editing

The source is in `src/`; `house.js` holds the floor plan (in feet).
After changing it:

```bash
npm install
npm run build   # writes dist/walkthrough.js
```
