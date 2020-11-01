from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import random
import os
import time
import numpy as np
executeOnCaeStartup()

inte_steps = 2000
impact_time = 0.008
sample_size = 2000
mesh_size = 4.0

DVs = []
for i in range(sample_size):
	rrr1 = random.uniform(6, 8)
	r2 = random.uniform(6, 8)
	r3 = random.uniform(6, 8)
	ddd1 = random.uniform(30, 40)
	d2 = random.uniform(80, 90)
	d3 = random.uniform(150, 160) 
	dvs = [rrr1, r2, r3, ddd1, d2, d3]
	# DVs.append(dvs)
	# np.save("G:/2. Backup/2. impact_V2.0/data_2D/objective functions/design_variables.npy", DVs)
	f_path = 'E:/1. impact/2D data/data/%d/' % int(i+468)
	if not os.path.exists(f_path):
		os.makedirs(f_path)
	f_path_1 = f_path + 'design_variables.txt'
	with open(f_path_1, 'a') as file_object:
		write_data = '%f %f %f %f %f %f \n' % (rrr1, r2, r3, ddd1, d2, d3)
		file_object.write(write_data)

	# rrr1 = 7.5
	# r2 = 7.5
	# r3 = 5
	# ddd1 = 25
	# d2 = 75
	# d3 = 135
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=200.0)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.Line(point1=(-56.25, 56.25), point2=(-32.5, 56.25))
	s.HorizontalConstraint(entity=g[2], addUndoState=False)
	s.Line(point1=(-32.5, 56.25), point2=(-18.75, 35.0))
	s.Line(point1=(-18.75, 35.0), point2=(-31.25, 15.0))
	s.Line(point1=(-31.25, 15.0), point2=(-55.0, 15.0))
	s.HorizontalConstraint(entity=g[5], addUndoState=False)
	s.Line(point1=(-55.0, 15.0), point2=(-70.0, 35.0))
	s.Line(point1=(-70.0, 35.0), point2=(-56.25, 56.25))
	s.ObliqueDimension(vertex1=v[0], vertex2=v[1], textPoint=(-51.0792274475098, 
		60.7162246704102), value=28.0)
	s.AngularDimension(line1=g[2], line2=g[3], textPoint=(-33.1843223571777, 
		49.3688735961914), value=120.0)
	s.ObliqueDimension(vertex1=v[1], vertex2=v[2], textPoint=(-16.0731353759766, 
		49.2384414672852), value=28.0)
	s.ObliqueDimension(vertex1=v[2], vertex2=v[3], textPoint=(-14.5057001113892, 
		22.1091499328613), value=28.0)
	s.AngularDimension(line1=g[3], line2=g[4], textPoint=(-21.9510192871094, 
		31.8913497924805), value=120.0)
	s.ObliqueDimension(vertex1=v[3], vertex2=v[4], textPoint=(-41.4133644104004, 
		2.54475402832031), value=28.0)
	s.ObliqueDimension(vertex1=v[4], vertex2=v[5], textPoint=(-70.5415725708008, 
		17.2832641601563), value=28.0)
	s.ObliqueDimension(vertex1=v[5], vertex2=v[0], textPoint=(-74.7214050292969, 
		45.9777069091797), value=28.0)
	s.copyMove(vector=(77.75, 1.74871130596426), objectList=(g[2], g[3], g[4], 
		g[5], g[6], g[7]))
	s.copyMove(vector=(-2.5, -62.5), objectList=(g[2], g[3], g[4], g[5], g[6], 
		g[7], g[8], g[9], g[10], g[11], g[12], g[13]))
	s.Line(point1=(-14.25, 32.0012886940357), point2=(7.5, 33.75))
	s.Line(point1=(-28.25, 7.75257738807145), point2=(-30.75, -6.25))
	s.Line(point1=(-16.75, -30.4987113059643), point2=(5.0, -28.75))
	s.Line(point1=(21.5, 9.50128869403571), point2=(19.0, -4.50128869403574))
	s.AngularDimension(line1=g[3], line2=g[26], textPoint=(-10.1952524185181, 
		36.3259429931641), value=120.0)
	s.AngularDimension(line1=g[5], line2=g[27], textPoint=(-32.5312271118164, 
		3.84904861450195), value=90.0)
	s.AngularDimension(line1=g[15], line2=g[28], textPoint=(-8.23595523834229, 
		-25.4975433349609), value=120.0)
	s.AngularDimension(line1=g[11], line2=g[29], textPoint=(24.9414749145508, 
		3.97947692871094), value=90.0)
	s.ObliqueDimension(vertex1=v[2], vertex2=v[11], textPoint=(-5.75417423248291, 
		42.5865478515625), value=28.0)
	s.ObliqueDimension(vertex1=v[3], vertex2=v[13], textPoint=(-26.9145736694336, 
		1.50131988525391), value=25.5)
	p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
		type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts['Part-1']
	p.BaseShellExtrude(sketch=s, depth=205.0)
	s.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts['Part-1']
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[26], sketchUpEdge=e[14], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-4.242052, 
		-35.845588, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=419.97, gridSpacing=10.49, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.Spot(point=(0.0, 70.8075))
	s1.Spot(point=(0.0, 18.3575))
	s1.Spot(point=(0.0, -39.3375))
	s1.CircleByCenterPerimeter(center=(0.0, 70.8075), point1=(2.6225, 60.3175))
	s1.CircleByCenterPerimeter(center=(0.0, 18.3575), point1=(2.6225, 10.49))
	s1.CircleByCenterPerimeter(center=(0.0, -39.3375), point1=(5.245, -49.8275))
	s1.DistanceDimension(entity1=v[8], entity2=g[10], textPoint=(92.9961413554688, 
		93.1195831298828), value=ddd1)
	s1.DistanceDimension(entity1=v[9], entity2=g[10], textPoint=(109.338579099609, 
		96.7459259033203), value=d2)
	s1.DistanceDimension(entity1=v[10], entity2=g[10], textPoint=(138.651140134766, 
		78.3551483154297), value=d3)
	s1.Line(point1=(-14.0000001526196, -102.5), point2=(-14.0000001526196, 
		-84.0536880493164))
	s1.VerticalConstraint(entity=g[17], addUndoState=False)
	s1.ParallelConstraint(entity1=g[8], entity2=g[17], addUndoState=False)
	s1.CoincidentConstraint(entity1=v[17], entity2=g[8], addUndoState=False)
	s1.Line(point1=(-14.0000001526196, -84.0536880493164), point2=(
		13.9999998474382, -84.0536880493164))
	s1.HorizontalConstraint(entity=g[18], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[17], entity2=g[18], addUndoState=False)
	s1.CoincidentConstraint(entity1=v[18], entity2=g[4], addUndoState=False)
	s1.Line(point1=(13.9999998474382, -84.0536880493164), point2=(13.9999998473804, 
		-102.5))
	s1.VerticalConstraint(entity=g[19], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[18], entity2=g[19], addUndoState=False)
	s1.Line(point1=(13.9999998473804, -102.5), point2=(-14.0000001526196, -102.5))
	s1.HorizontalConstraint(entity=g[20], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[19], entity2=g[20], addUndoState=False)
	s1.DistanceDimension(entity1=g[12], entity2=g[18], textPoint=(98.7030200175781, 
		-86.6439361572266), value=20.0)
	s1.RadialDimension(curve=g[16], textPoint=(22.4384001903076, 
		-29.9174118041992), radius=rrr1)
	s1.RadialDimension(curve=g[15], textPoint=(21.4007872752686, 36.9111022949219), 
		radius=r2)
	s1.RadialDimension(curve=g[14], textPoint=(21.9196013621826, 83.5356597900391), 
		radius=r3)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[26], sketchUpEdge=e1[14], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[1], sketchUpEdge=e[11], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-4.242052, 
		-35.845588, 112.024946))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=413.8, gridSpacing=10.34, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(-1.52619605842119e-07, 92.975054), point2=(85.305, 
		-116.325))
	s.CoincidentConstraint(entity1=v[16], entity2=g[7], addUndoState=False)
	s.EqualDistanceConstraint(entity1=v[8], entity2=v[9], midpoint=v[16], 
		addUndoState=False)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[1], sketchUpEdge=e1[11], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[16], sketchUpEdge=e[65], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(37.757948, 
		-60.0943, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=435.94, gridSpacing=10.89, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.rectangle(point1=(-35.3925, 103.455), point2=(49.005, -106.1775))
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[16], sketchUpEdge=e1[65], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[14], sketchUpEdge=e[41], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-32.242052, 
		1.153123, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=467.14, gridSpacing=11.67, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(1.09619961108365e-07, 102.5), point2=(134.205, -105.03))
	s.CoincidentConstraint(entity1=v[8], entity2=g[10], addUndoState=False)
	s.EqualDistanceConstraint(entity1=v[5], entity2=v[2], midpoint=v[8], 
		addUndoState=False)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[14], sketchUpEdge=e1[41], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=443.91, gridSpacing=11.09, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f1[10], sketchUpEdge=e1[37], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-23.721233, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=430.18, gridSpacing=10.75, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(-29.5625, 104.8125), point2=(56.4375, -104.8125))
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	p.CutExtrude(sketchPlane=f[10], sketchUpEdge=e[37], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f1[9], sketchUpEdge=e1[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=422.24, gridSpacing=10.55, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=422.24, gridSpacing=10.55, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1 = p.faces
	p.RemoveFaces(faceList = f1[8:11], deleteCells=False)
	mdb.models['Model-1'].Material(name='Material-1')
	mdb.models['Model-1'].materials['Material-1'].Density(table=((2.7e-09, ), ))
	mdb.models['Model-1'].materials['Material-1'].Elastic(table=((72000.0, 0.33), 
		))
	mdb.models['Model-1'].materials['Material-1'].Plastic(table=((131.8, 0.0), (
		140.0, 0.0023), (149.4, 0.0067), (162.3, 0.0153), (171.2, 0.0232), (178.3, 
		0.0314), (183.7, 0.0406), (187.4, 0.0503), (189.8, 0.0601), (190.6, 
		0.0648), (191.0, 0.0658)))
	mdb.models['Model-1'].HomogeneousShellSection(name='Section-1', 
		preIntegrate=OFF, material='Material-1', thicknessType=UNIFORM, 
		thickness=2.0, thicknessField='', nodalThicknessField='', 
		idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
		thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
		integrationRule=SIMPSON, numIntPts=5)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
	region = p.Set(faces=faces, name='Set-1')
	p = mdb.models['Model-1'].parts['Part-1']
	p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', 
		thicknessAssignment=FROM_SECTION)
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=200.0)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.rectangle(point1=(-68.75, 61.25), point2=(73.75, -60.0))
	p = mdb.models['Model-1'].Part(name='Part-2', dimensionality=THREE_D, 
		type=DISCRETE_RIGID_SURFACE)
	p = mdb.models['Model-1'].parts['Part-2']
	p.BaseShell(sketch=s)
	s.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts['Part-2']
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-2']
	e = p.edges
	p.DatumPointByMidPoint(point1=p.InterestingPoint(edge=e[1], rule=MIDDLE), 
		point2=p.InterestingPoint(edge=e[3], rule=MIDDLE))
	p = mdb.models['Model-1'].parts['Part-2']
	v1, e1, d1, n = p.vertices, p.edges, p.datums, p.nodes
	p.ReferencePoint(point=d1[2])
	a = mdb.models['Model-1'].rootAssembly
	a.DatumCsysByDefault(CARTESIAN)
	p = mdb.models['Model-1'].parts['Part-1']
	a.Instance(name='Part-1-1', part=p, dependent=ON)
	p = mdb.models['Model-1'].parts['Part-2']
	a.Instance(name='Part-2-1', part=p, dependent=ON)
	p = mdb.models['Model-1'].parts['Part-1']
	e1 = p.edges
	p.DatumPointByMidPoint(point1=p.InterestingPoint(edge=e1[18], rule=MIDDLE), 
		point2=p.InterestingPoint(edge=e1[26], rule=MIDDLE))
	p = mdb.models['Model-1'].parts['Part-1']
	v1, e, d1, n1 = p.vertices, p.edges, p.datums, p.nodes
	p.ReferencePoint(point=d1[9])
	p = mdb.models['Model-1'].parts['Part-1']
	r = p.referencePoints
	refPoints=(r[10], )
	region=p.Set(referencePoints=refPoints, name='Set-2')
	mdb.models['Model-1'].parts['Part-1'].engineeringFeatures.PointMassInertia(
		name='Inertia-1', region=region, mass=2, alpha=0.0, composite=0.0)
	a = mdb.models['Model-1'].rootAssembly
	a.regenerate()
	a = mdb.models['Model-1'].rootAssembly
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-2-1'].referencePoints
	a.DatumPointByOffset(point=r1[3], vector=(0.0, 0.0, 210.0))
	a1 = mdb.models['Model-1'].rootAssembly
	a1.translate(instanceList=('Part-1-1', ), vector=(48.742052, -37.526834, 5.0))
	mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
		timePeriod=impact_time, improvedDtMethod=ON)
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
		numIntervals=inte_steps)
	mdb.models['Model-1'].ContactProperty('IntProp-1')
	mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
		formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
		pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
		0.17, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
		fraction=0.005, elasticSlipStiffness=None)
	#: The interaction property "IntProp-1" has been created.
	mdb.models['Model-1'].ContactExp(name='Int-1', createStepName='Step-1')
	mdb.models['Model-1'].interactions['Int-1'].includedPairs.setValuesInStep(
		stepName='Step-1', useAllstar=ON)
	mdb.models['Model-1'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
		stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-1-1'].referencePoints
	refPoints1=(r1[10], )
	region1=a.Set(referencePoints=refPoints1, name='m_Set-1')
	a = mdb.models['Model-1'].rootAssembly
	s1 = a.instances['Part-1-1'].edges
	side1Edges1 = s1.getSequenceFromMask(mask=('[#24a44802 ]', ), )
	region2=a.Surface(side1Edges=side1Edges1, name='s_Surf-1')
	mdb.models['Model-1'].Coupling(name='Constraint-1', controlPoint=region1, 
		surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
		localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
	a = mdb.models['Model-1'].rootAssembly
	f1 = a.instances['Part-1-1'].faces
	faces1 = f1.getSequenceFromMask(mask=('[#ff ]', ), )
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#2da6df87 ]', ), )
	v1 = a.instances['Part-1-1'].vertices
	verts1 = v1.getSequenceFromMask(mask=('[#28df06 ]', ), )
	r1 = a.instances['Part-1-1'].referencePoints
	refPoints1=(r1[10], )
	region = a.Set(vertices=verts1, edges=edges1, faces=faces1, 
		referencePoints=refPoints1, name='Set-2')
	mdb.models['Model-1'].Velocity(name='Predefined Field-1', region=region, 
		field='', distributionType=MAGNITUDE, velocity1=0.0, velocity2=0.0, 
		velocity3=-15510.0, omega=0.0)
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-2-1'].referencePoints
	refPoints1=(r1[3], )
	region = a.Set(referencePoints=refPoints1, name='Set-3')
	mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
		region=region, u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET, 
		amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
	a = mdb.models['Model-1'].rootAssembly
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#1 ]', ), )
	region = a.Set(edges=edges1, name='Set-4')
	mdb.models['Model-1'].YsymmBC(name='BC-2', createStepName='Initial', 
		region=region, localCsys=None)
	a = mdb.models['Model-1'].rootAssembly
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#550 ]', ), )
	region = a.Set(edges=edges1, name='Set-5')
	mdb.models['Model-1'].XsymmBC(name='BC-3', createStepName='Initial', 
		region=region, localCsys=None)
	p = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
	p = mdb.models['Model-1'].parts['Part-1']
	p.generateMesh()
	p = mdb.models['Model-1'].parts['Part-2']
	p = mdb.models['Model-1'].parts['Part-2']
	p.seedPart(size=14.0, deviationFactor=0.1, minSizeFactor=0.1)
	p = mdb.models['Model-1'].parts['Part-2']
	p.generateMesh()
	elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
	elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
	p = mdb.models['Model-1'].parts['Part-2']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
	pickedRegions =(faces, )
	p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
	p = mdb.models['Model-1'].parts['Part-1']
	elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT, 
		secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
	elemType2 = mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
	pickedRegions =(faces, )
	p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))

	a = mdb.models['Model-1'].rootAssembly
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
	    variables=PRESELECT)
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
	    'S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A', 
	    'RF', 'CSTRESS', 'ENER', 'ELEN', 'ELEDEN', 'EDCDEN', 'EDT', 'EVF'))
	mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
	    'ALLAE', 'ALLCD', 'ALLDC', 'ALLDMD', 'ALLFD', 'ALLIE', 'ALLKE', 'ALLPD', 
	    'ALLSE', 'ALLVD', 'ALLWK', 'ALLCW', 'ALLMW', 'ALLPW', 'ETOTAL'))


	myjob = mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
		atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
		memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
		nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
		contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
		resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=4, 
		activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=4)
	mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
	myjob.waitForCompletion()
	#: The job input file "Job-1.inp" has been submitted for analysis.
	#: Job Job-1: Analysis Input File Processor completed successfully.
	#: Job Job-1: Abaqus/Explicit Packager completed successfully.
	o3 = session.openOdb(name='C:/temp/Job-1.odb')

	from odbAccess import *
	odb = openOdb(path='C:/temp/Job-1.odb')
	scratchOdb = session.ScratchOdb(odb)
	session.viewports['Viewport: 1'].setValues(displayedObject=odb)
	scratchOdb = session.scratchOdbs['C:/temp/Job-1.odb']

	scratchOdb.rootAssembly.DatumCsysByThreePoints(name='CSYS-1', 
		coordSysType=CARTESIAN, origin=(44.5, -36.3737, 210.0), point1=(44.5, 
		0.62499862909317, 210.0), point2=(16.4999980926514, -36.3737106323242, 
		210.0))

	session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
		mirrorCsysName='CSYS-1', mirrorAboutYzPlane=True, mirrorAboutXzPlane=True)


	odb = session.odbs['C:/temp/Job-1.odb']
	session.viewports['Viewport: 1'].setValues(displayedObject=odb)
	session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
		CONTOURS_ON_DEF, ))
	session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
	session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
	leaf = dgo.LeafFromElementSets(elementSets=("PART-2-1.PART-2", ))
	session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)
	leaf = dgo.LeafFromUserCoordSystem(coordSystems=("CSYS-1", ))
	session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)
	# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
	session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
	session.graphicsOptions.setValues(backgroundColor='#FFFFFF', 
		backgroundBottomColor='#FFFFFF')
	session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
		variableLabel='A', outputPosition=NODAL, refinement=(INVARIANT, 
		'Magnitude'), )
	session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
		legendMinMax=ON, title=OFF, state=OFF, annotations=OFF, compass=OFF)
	session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
		outsideLimitsAboveColor='#0000FF', outsideLimitsBelowColor='#FF0000', 
		maxAutoCompute=OFF, maxValue=-1, minAutoCompute=OFF, minValue=-2)
	session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
		outsideLimitsAboveColor='#0000FF', outsideLimitsBelowColor='#FF0000')
	session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
		visibleEdges=NONE)
	leaf = dgo.LeafFromSurfaceSets(surfaceSets=("Rigid Reference Node #2629", ))
	session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)

	for step in range(inte_steps):
		session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=step+1)
		session.printOptions.setValues(vpBackground=ON)
		# fileName1 = 'C:/2.backup/1.impact/training_data_acce/impact_image/%d/' % int(i+1181)
		fileName1 = 'E:/1. impact/2D data/images/%d/' % int(i+468)
		if not os.path.exists(fileName1):
			os.makedirs(fileName1)
		fileName1 = fileName1 + '%d' % step
		session.printToFile(fileName=fileName1, 
		    format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

	odb = openOdb(path='C:/temp/Job-1.odb')
	# file_object.write(str(rrr1)+' '+str(r2)+' '+str(r3)+' '+str(ddd1)+' '+str(d2)+' '+str(d3)+'\n')
	# print(odb.steps['Step-1'].frames[1].fieldOutputs.keys())
	# print(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs.keys())
	# print(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLAE'].data)
	f_path_2 = f_path + 'field_ofs.txt'
	with open(f_path_2, 'a') as file_object:
		for step in range(inte_steps):
			RF = odb.steps['Step-1'].frames[step+1].fieldOutputs['RF']
			RFValues = RF.values
			RF = []
			for rf in RFValues:
				rf = rf.data
				# v_ = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
				rf_ = rf[2]
				RF.append(rf_)
				# print(v)
			rf_max = max(RF)

			A = odb.steps['Step-1'].frames[step+1].fieldOutputs['A']
			AValues = A.values
			A = []
			for a in AValues:
				a = a.data
				a_ = a[2]
				A.append(a_)
			a_max = max(A)

			SValues = odb.steps['Step-1'].frames[step+1].fieldOutputs['S'].values[0].mises
			# S = []
			# for s in SValues:
			# 	s = s.data
			# 	# print(s)
			# 	s_ = s
			# 	S.append(s_)
			# s_max = max(S)
			
			write_data = '%f %f %f \n' % (rf_max, a_max, SValues)
			file_object.write(write_data)

	f_path_3 = f_path + 'history_ofs.txt'
	with open(f_path_3, 'a') as file_object:
		time = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLAE'].data)[:, 0]
		for d in range(len(time)):
			time = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLAE'].data)[:, 0][d]
			allea = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLAE'].data)[:, 1][d]
			allcd = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLCD'].data)[:, 1][d]
			allcw = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLCW'].data)[:, 1][d]
			alldmd = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLDMD'].data)[:, 1][d]
			allfd = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLFD'].data)[:, 1][d]
			allie = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLIE'].data)[:, 1][d]
			allka = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLKE'].data)[:, 1][d]
			allmw = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLMW'].data)[:, 1][d]
			allpw = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLPW'].data)[:, 1][d]
			allse = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLSE'].data)[:, 1][d]
			allvd = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLVD'].data)[:, 1][d]
			allwk = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data)[:, 1][d]
			etotal = np.array(odb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ETOTAL'].data)[:, 1][d]
			OFs = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f \n' % (
				time, allea, allcd, allcw, alldmd, allfd, allie, allka, allmw, allpw, allse, allvd, allwk, etotal)
			file_object.write(OFs)
