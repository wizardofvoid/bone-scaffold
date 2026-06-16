import adsk.core, adsk.fusion, adsk.cam, traceback
import math

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        
        if not design:
            ui.messageBox('Please create a new design first.', 'No Active Design')
            return

        # ==============================================================================
        # Scaffold Unit Cell Parameters
        # ==============================================================================
        # These correspond to the variables `c` and `p` from your `problem.py`
        c_mm = 5.0     # Unit cell size (c) in mm (Bounds: 1.0 to 5.0)
        p = 0.60       # Porosity fraction (p) (Bounds: 0.5 to 0.9)
        
        # Fusion 360's internal API unit is always centimeters (cm).
        # We must convert our parameters from mm to cm.
        c = c_mm / 10.0 
        
        # Calculate pore radius in cm: V_pore = p * c^3 = 4/3 * pi * r_pore^3
        r_pore = c * ((3.0 * p) / (4.0 * math.pi)) ** (1.0 / 3.0)
        # ==============================================================================

        # Get root component
        rootComp = design.rootComponent

        # Get TemporaryBRepManager for fast, silent solid operations
        tempBrepMgr = adsk.fusion.TemporaryBRepManager.get()
        
        # 1. Create a Temporary Box (Block)
        # The block is centered at c/2, c/2, c/2 with size c x c x c to match MAPDL logic
        center = adsk.core.Point3D.create(c/2.0, c/2.0, c/2.0)
        lengthDir = adsk.core.Vector3D.create(1, 0, 0)
        widthDir = adsk.core.Vector3D.create(0, 1, 0)
        
        # OrientedBoundingBox3D needs center, lengthDir, widthDir, length, width, height
        boxBnd = adsk.core.OrientedBoundingBox3D.create(center, lengthDir, widthDir, c, c, c)
        blockBody = tempBrepMgr.createBox(boxBnd)

        # 2. Create a Temporary Sphere (Pore)
        sphereBody = tempBrepMgr.createSphere(center, r_pore)

        # 3. Subtract the Sphere from the Block
        # This modifies blockBody in place (Difference = Cut)
        tempBrepMgr.booleanOperation(blockBody, sphereBody, adsk.fusion.BooleanTypes.DifferenceBooleanType)

        # 4. Add the resulting body into the Fusion 360 design
        # We use a BaseFeature to hold the direct modeled body
        baseFeatures = rootComp.features.baseFeatures
        baseFeature = baseFeatures.add()
        
        baseFeature.startEdit()
        body = rootComp.bRepBodies.add(blockBody, baseFeature)
        body.name = f"Scaffold_Cell_{c_mm}mm_{int(p*100)}p"
        baseFeature.finishEdit()

        ui.messageBox(f'Successfully generated scaffold cell!\nSize: {c_mm}mm\nPorosity: {p*100}%\nPore Radius: {r_pore*10.0:.2f}mm')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
