??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??
?
Adam/common.color/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*)
shared_nameAdam/common.color/bias/v
?
,Adam/common.color/bias/v/Read/ReadVariableOpReadVariableOpAdam/common.color/bias/v*
_output_shapes
:!*
dtype0
?
Adam/common.color/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*+
shared_nameAdam/common.color/kernel/v
?
.Adam/common.color/kernel/v/Read/ReadVariableOpReadVariableOpAdam/common.color/kernel/v*
_output_shapes
:	?!*
dtype0
?
 Adam/common.apparel_class/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/common.apparel_class/bias/v
?
4Adam/common.apparel_class/bias/v/Read/ReadVariableOpReadVariableOp Adam/common.apparel_class/bias/v*
_output_shapes
:*
dtype0
?
"Adam/common.apparel_class/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/common.apparel_class/kernel/v
?
6Adam/common.apparel_class/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/common.apparel_class/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_main_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/dense_main_last/bias/v
?
/Adam/dense_main_last/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/dense_main_last/kernel/v
?
1Adam/dense_main_last/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_main_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/dense_main_2/bias/v
?
,Adam/dense_main_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/dense_main_2/kernel/v
?
.Adam/dense_main_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_main_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/dense_main_1/bias/v
?
,Adam/dense_main_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*+
shared_nameAdam/dense_main_1/kernel/v
?
.Adam/dense_main_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/kernel/v* 
_output_shapes
:
?@?*
dtype0
?
Adam/conv_main_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_main_3/bias/v

+Adam/conv_main_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv_main_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_main_3/kernel/v
?
-Adam/conv_main_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv_main_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_main_2/bias/v

+Adam/conv_main_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_main_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_2/kernel/v
?
-Adam/conv_main_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv_main_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_main_1/bias/v

+Adam/conv_main_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_main_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_1/kernel/v
?
-Adam/conv_main_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/common.color/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*)
shared_nameAdam/common.color/bias/m
?
,Adam/common.color/bias/m/Read/ReadVariableOpReadVariableOpAdam/common.color/bias/m*
_output_shapes
:!*
dtype0
?
Adam/common.color/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*+
shared_nameAdam/common.color/kernel/m
?
.Adam/common.color/kernel/m/Read/ReadVariableOpReadVariableOpAdam/common.color/kernel/m*
_output_shapes
:	?!*
dtype0
?
 Adam/common.apparel_class/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/common.apparel_class/bias/m
?
4Adam/common.apparel_class/bias/m/Read/ReadVariableOpReadVariableOp Adam/common.apparel_class/bias/m*
_output_shapes
:*
dtype0
?
"Adam/common.apparel_class/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/common.apparel_class/kernel/m
?
6Adam/common.apparel_class/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/common.apparel_class/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_main_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdam/dense_main_last/bias/m
?
/Adam/dense_main_last/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdam/dense_main_last/kernel/m
?
1Adam/dense_main_last/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_main_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/dense_main_2/bias/m
?
,Adam/dense_main_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/dense_main_2/kernel/m
?
.Adam/dense_main_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_main_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/dense_main_1/bias/m
?
,Adam/dense_main_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_main_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*+
shared_nameAdam/dense_main_1/kernel/m
?
.Adam/dense_main_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/kernel/m* 
_output_shapes
:
?@?*
dtype0
?
Adam/conv_main_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_main_3/bias/m

+Adam/conv_main_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv_main_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_main_3/kernel/m
?
-Adam/conv_main_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv_main_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_main_2/bias/m

+Adam/conv_main_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_main_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_2/kernel/m
?
-Adam/conv_main_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv_main_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_main_1/bias/m

+Adam/conv_main_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_main_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_1/kernel/m
?
-Adam/conv_main_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_1/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
z
common.color/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*"
shared_namecommon.color/bias
s
%common.color/bias/Read/ReadVariableOpReadVariableOpcommon.color/bias*
_output_shapes
:!*
dtype0
?
common.color/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*$
shared_namecommon.color/kernel
|
'common.color/kernel/Read/ReadVariableOpReadVariableOpcommon.color/kernel*
_output_shapes
:	?!*
dtype0
?
common.apparel_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecommon.apparel_class/bias
?
-common.apparel_class/bias/Read/ReadVariableOpReadVariableOpcommon.apparel_class/bias*
_output_shapes
:*
dtype0
?
common.apparel_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namecommon.apparel_class/kernel
?
/common.apparel_class/kernel/Read/ReadVariableOpReadVariableOpcommon.apparel_class/kernel*
_output_shapes
:	?*
dtype0
?
dense_main_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namedense_main_last/bias
z
(dense_main_last/bias/Read/ReadVariableOpReadVariableOpdense_main_last/bias*
_output_shapes	
:?*
dtype0
?
dense_main_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_namedense_main_last/kernel
?
*dense_main_last/kernel/Read/ReadVariableOpReadVariableOpdense_main_last/kernel* 
_output_shapes
:
??*
dtype0
{
dense_main_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namedense_main_2/bias
t
%dense_main_2/bias/Read/ReadVariableOpReadVariableOpdense_main_2/bias*
_output_shapes	
:?*
dtype0
?
dense_main_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namedense_main_2/kernel
}
'dense_main_2/kernel/Read/ReadVariableOpReadVariableOpdense_main_2/kernel* 
_output_shapes
:
??*
dtype0
{
dense_main_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namedense_main_1/bias
t
%dense_main_1/bias/Read/ReadVariableOpReadVariableOpdense_main_1/bias*
_output_shapes	
:?*
dtype0
?
dense_main_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@?*$
shared_namedense_main_1/kernel
}
'dense_main_1/kernel/Read/ReadVariableOpReadVariableOpdense_main_1/kernel* 
_output_shapes
:
?@?*
dtype0
x
conv_main_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv_main_3/bias
q
$conv_main_3/bias/Read/ReadVariableOpReadVariableOpconv_main_3/bias*
_output_shapes
: *
dtype0
?
conv_main_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv_main_3/kernel
?
&conv_main_3/kernel/Read/ReadVariableOpReadVariableOpconv_main_3/kernel*&
_output_shapes
: *
dtype0
x
conv_main_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_main_2/bias
q
$conv_main_2/bias/Read/ReadVariableOpReadVariableOpconv_main_2/bias*
_output_shapes
:*
dtype0
?
conv_main_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_main_2/kernel
?
&conv_main_2/kernel/Read/ReadVariableOpReadVariableOpconv_main_2/kernel*&
_output_shapes
:*
dtype0
x
conv_main_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_main_1/bias
q
$conv_main_1/bias/Read/ReadVariableOpReadVariableOpconv_main_1/bias*
_output_shapes
:*
dtype0
?
conv_main_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_main_1/kernel
?
&conv_main_1/kernel/Read/ReadVariableOpReadVariableOpconv_main_1/kernel*&
_output_shapes
:*
dtype0
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv_main_1/kernelconv_main_1/biasconv_main_2/kernelconv_main_2/biasconv_main_3/kernelconv_main_3/biasdense_main_1/kerneldense_main_1/biasdense_main_2/kerneldense_main_2/biasdense_main_last/kerneldense_main_last/biascommon.color/kernelcommon.color/biascommon.apparel_class/kernelcommon.apparel_class/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_44084

NoOpNoOp
ڀ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op*
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op*
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator* 
?
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias*
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias*
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias*
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias*
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias*
z
0
1
-2
.3
<4
=5
R6
S7
Z8
[9
b10
c11
j12
k13
r14
s15*
z
0
1
-2
.3
<4
=5
R6
S7
Z8
[9
b10
c11
j12
k13
r14
s15*
* 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ytrace_0
ztrace_1
{trace_2
|trace_3* 
7
}trace_0
~trace_1
trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?-m?.m?<m?=m?Rm?Sm?Zm?[m?bm?cm?jm?km?rm?sm?v?v?-v?.v?<v?=v?Rv?Sv?Zv?[v?bv?cv?jv?kv?rv?sv?*
* 

?serving_default* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
b\
VARIABLE_VALUEconv_main_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

-0
.1*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
b\
VARIABLE_VALUEconv_main_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

<0
=1*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
b\
VARIABLE_VALUEconv_main_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

R0
S1*

R0
S1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
c]
VARIABLE_VALUEdense_main_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdense_main_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
c]
VARIABLE_VALUEdense_main_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdense_main_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

b0
c1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
f`
VARIABLE_VALUEdense_main_last/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEdense_main_last/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

j0
k1*

j0
k1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEcommon.apparel_class/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcommon.apparel_class/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

r0
s1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
c]
VARIABLE_VALUEcommon.color/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcommon.color/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*
,
?0
?1
?2
?3
?4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
<
?	variables
?	keras_api

?total

?count*
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
VARIABLE_VALUEAdam/conv_main_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv_main_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv_main_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_main_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_main_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_last/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_main_last/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/common.apparel_class/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/common.apparel_class/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/common.color/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/common.color/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv_main_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv_main_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv_main_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv_main_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_main_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_main_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_main_last/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_main_last/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/common.apparel_class/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/common.apparel_class/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/common.color/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/common.color/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&conv_main_1/kernel/Read/ReadVariableOp$conv_main_1/bias/Read/ReadVariableOp&conv_main_2/kernel/Read/ReadVariableOp$conv_main_2/bias/Read/ReadVariableOp&conv_main_3/kernel/Read/ReadVariableOp$conv_main_3/bias/Read/ReadVariableOp'dense_main_1/kernel/Read/ReadVariableOp%dense_main_1/bias/Read/ReadVariableOp'dense_main_2/kernel/Read/ReadVariableOp%dense_main_2/bias/Read/ReadVariableOp*dense_main_last/kernel/Read/ReadVariableOp(dense_main_last/bias/Read/ReadVariableOp/common.apparel_class/kernel/Read/ReadVariableOp-common.apparel_class/bias/Read/ReadVariableOp'common.color/kernel/Read/ReadVariableOp%common.color/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/conv_main_1/kernel/m/Read/ReadVariableOp+Adam/conv_main_1/bias/m/Read/ReadVariableOp-Adam/conv_main_2/kernel/m/Read/ReadVariableOp+Adam/conv_main_2/bias/m/Read/ReadVariableOp-Adam/conv_main_3/kernel/m/Read/ReadVariableOp+Adam/conv_main_3/bias/m/Read/ReadVariableOp.Adam/dense_main_1/kernel/m/Read/ReadVariableOp,Adam/dense_main_1/bias/m/Read/ReadVariableOp.Adam/dense_main_2/kernel/m/Read/ReadVariableOp,Adam/dense_main_2/bias/m/Read/ReadVariableOp1Adam/dense_main_last/kernel/m/Read/ReadVariableOp/Adam/dense_main_last/bias/m/Read/ReadVariableOp6Adam/common.apparel_class/kernel/m/Read/ReadVariableOp4Adam/common.apparel_class/bias/m/Read/ReadVariableOp.Adam/common.color/kernel/m/Read/ReadVariableOp,Adam/common.color/bias/m/Read/ReadVariableOp-Adam/conv_main_1/kernel/v/Read/ReadVariableOp+Adam/conv_main_1/bias/v/Read/ReadVariableOp-Adam/conv_main_2/kernel/v/Read/ReadVariableOp+Adam/conv_main_2/bias/v/Read/ReadVariableOp-Adam/conv_main_3/kernel/v/Read/ReadVariableOp+Adam/conv_main_3/bias/v/Read/ReadVariableOp.Adam/dense_main_1/kernel/v/Read/ReadVariableOp,Adam/dense_main_1/bias/v/Read/ReadVariableOp.Adam/dense_main_2/kernel/v/Read/ReadVariableOp,Adam/dense_main_2/bias/v/Read/ReadVariableOp1Adam/dense_main_last/kernel/v/Read/ReadVariableOp/Adam/dense_main_last/bias/v/Read/ReadVariableOp6Adam/common.apparel_class/kernel/v/Read/ReadVariableOp4Adam/common.apparel_class/bias/v/Read/ReadVariableOp.Adam/common.color/kernel/v/Read/ReadVariableOp,Adam/common.color/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_44732
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_main_1/kernelconv_main_1/biasconv_main_2/kernelconv_main_2/biasconv_main_3/kernelconv_main_3/biasdense_main_1/kerneldense_main_1/biasdense_main_2/kerneldense_main_2/biasdense_main_last/kerneldense_main_last/biascommon.apparel_class/kernelcommon.apparel_class/biascommon.color/kernelcommon.color/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_4count_4total_3count_3total_2count_2total_1count_1totalcountAdam/conv_main_1/kernel/mAdam/conv_main_1/bias/mAdam/conv_main_2/kernel/mAdam/conv_main_2/bias/mAdam/conv_main_3/kernel/mAdam/conv_main_3/bias/mAdam/dense_main_1/kernel/mAdam/dense_main_1/bias/mAdam/dense_main_2/kernel/mAdam/dense_main_2/bias/mAdam/dense_main_last/kernel/mAdam/dense_main_last/bias/m"Adam/common.apparel_class/kernel/m Adam/common.apparel_class/bias/mAdam/common.color/kernel/mAdam/common.color/bias/mAdam/conv_main_1/kernel/vAdam/conv_main_1/bias/vAdam/conv_main_2/kernel/vAdam/conv_main_2/bias/vAdam/conv_main_3/kernel/vAdam/conv_main_3/bias/vAdam/dense_main_1/kernel/vAdam/dense_main_1/bias/vAdam/dense_main_2/kernel/vAdam/dense_main_2/bias/vAdam/dense_main_last/kernel/vAdam/dense_main_last/bias/v"Adam/common.apparel_class/kernel/v Adam/common.apparel_class/bias/vAdam/common.color/kernel/vAdam/common.color/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_44931??	
?:
?
B__inference_model_1_layer_call_and_return_conditional_losses_43988	
input+
conv_main_1_43942:
conv_main_1_43944:+
conv_main_2_43948:
conv_main_2_43950:+
conv_main_3_43954: 
conv_main_3_43956: &
dense_main_1_43961:
?@?!
dense_main_1_43963:	?&
dense_main_2_43966:
??!
dense_main_2_43968:	?)
dense_main_last_43971:
??$
dense_main_last_43973:	?%
common_color_43976:	?! 
common_color_43978:!-
common_apparel_class_43981:	?(
common_apparel_class_43983:
identity

identity_1??,common.apparel_class/StatefulPartitionedCall?$common.color/StatefulPartitionedCall?#conv_main_1/StatefulPartitionedCall?#conv_main_2/StatefulPartitionedCall?#conv_main_3/StatefulPartitionedCall?$dense_main_1/StatefulPartitionedCall?$dense_main_2/StatefulPartitionedCall?'dense_main_last/StatefulPartitionedCall?
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputconv_main_1_43942conv_main_1_43944*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483?
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450?
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_43948conv_main_2_43950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501?
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462?
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_43954conv_main_3_43956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519?
flatten_1/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531?
dropout_main/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43538?
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall%dropout_main/PartitionedCall:output:0dense_main_1_43961dense_main_1_43963*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551?
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_43966dense_main_2_43968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568?
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_43971dense_main_last_43973*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585?
$common.color/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_color_43976common_color_43978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_common.color_layer_call_and_return_conditional_losses_43602?
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_43981common_apparel_class_43983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619?
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-common.color/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall%^common.color/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2L
$common.color/StatefulPartitionedCall$common.color/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?U
?
B__inference_model_1_layer_call_and_return_conditional_losses_44228

inputsD
*conv_main_1_conv2d_readvariableop_resource:9
+conv_main_1_biasadd_readvariableop_resource:D
*conv_main_2_conv2d_readvariableop_resource:9
+conv_main_2_biasadd_readvariableop_resource:D
*conv_main_3_conv2d_readvariableop_resource: 9
+conv_main_3_biasadd_readvariableop_resource: ?
+dense_main_1_matmul_readvariableop_resource:
?@?;
,dense_main_1_biasadd_readvariableop_resource:	??
+dense_main_2_matmul_readvariableop_resource:
??;
,dense_main_2_biasadd_readvariableop_resource:	?B
.dense_main_last_matmul_readvariableop_resource:
??>
/dense_main_last_biasadd_readvariableop_resource:	?>
+common_color_matmul_readvariableop_resource:	?!:
,common_color_biasadd_readvariableop_resource:!F
3common_apparel_class_matmul_readvariableop_resource:	?B
4common_apparel_class_biasadd_readvariableop_resource:
identity

identity_1??+common.apparel_class/BiasAdd/ReadVariableOp?*common.apparel_class/MatMul/ReadVariableOp?#common.color/BiasAdd/ReadVariableOp?"common.color/MatMul/ReadVariableOp?"conv_main_1/BiasAdd/ReadVariableOp?!conv_main_1/Conv2D/ReadVariableOp?"conv_main_2/BiasAdd/ReadVariableOp?!conv_main_2/Conv2D/ReadVariableOp?"conv_main_3/BiasAdd/ReadVariableOp?!conv_main_3/Conv2D/ReadVariableOp?#dense_main_1/BiasAdd/ReadVariableOp?"dense_main_1/MatMul/ReadVariableOp?#dense_main_2/BiasAdd/ReadVariableOp?"dense_main_2/MatMul/ReadVariableOp?&dense_main_last/BiasAdd/ReadVariableOp?%dense_main_last/MatMul/ReadVariableOp?
!conv_main_1/Conv2D/ReadVariableOpReadVariableOp*conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_main_1/Conv2DConv2Dinputs)conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
"conv_main_1/BiasAdd/ReadVariableOpReadVariableOp+conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_main_1/BiasAddBiasAddconv_main_1/Conv2D:output:0*conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????r
conv_main_1/ReluReluconv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
maxpool_main_1/MaxPoolMaxPoolconv_main_1/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
?
!conv_main_2/Conv2D/ReadVariableOpReadVariableOp*conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_main_2/Conv2DConv2Dmaxpool_main_1/MaxPool:output:0)conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
"conv_main_2/BiasAdd/ReadVariableOpReadVariableOp+conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_main_2/BiasAddBiasAddconv_main_2/Conv2D:output:0*conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@p
conv_main_2/ReluReluconv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
maxpool_main_2/MaxPoolMaxPoolconv_main_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
!conv_main_3/Conv2D/ReadVariableOpReadVariableOp*conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv_main_3/Conv2DConv2Dmaxpool_main_2/MaxPool:output:0)conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
"conv_main_3/BiasAdd/ReadVariableOpReadVariableOp+conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv_main_3/BiasAddBiasAddconv_main_3/Conv2D:output:0*conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? p
conv_main_3/ReluReluconv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten_1/ReshapeReshapeconv_main_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????@p
dropout_main/IdentityIdentityflatten_1/Reshape:output:0*
T0*(
_output_shapes
:??????????@?
"dense_main_1/MatMul/ReadVariableOpReadVariableOp+dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
?@?*
dtype0?
dense_main_1/MatMulMatMuldropout_main/Identity:output:0*dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#dense_main_1/BiasAdd/ReadVariableOpReadVariableOp,dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_1/BiasAddBiasAdddense_main_1/MatMul:product:0+dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
dense_main_1/ReluReludense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"dense_main_2/MatMul/ReadVariableOpReadVariableOp+dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_main_2/MatMulMatMuldense_main_1/Relu:activations:0*dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#dense_main_2/BiasAdd/ReadVariableOpReadVariableOp,dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_2/BiasAddBiasAdddense_main_2/MatMul:product:0+dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
dense_main_2/ReluReludense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
%dense_main_last/MatMul/ReadVariableOpReadVariableOp.dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_main_last/MatMulMatMuldense_main_2/Relu:activations:0-dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&dense_main_last/BiasAdd/ReadVariableOpReadVariableOp/dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_last/BiasAddBiasAdd dense_main_last/MatMul:product:0.dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
dense_main_last/TanhTanh dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"common.color/MatMul/ReadVariableOpReadVariableOp+common_color_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype0?
common.color/MatMulMatMuldense_main_last/Tanh:y:0*common.color/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!?
#common.color/BiasAdd/ReadVariableOpReadVariableOp,common_color_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype0?
common.color/BiasAddBiasAddcommon.color/MatMul:product:0+common.color/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!p
common.color/SoftmaxSoftmaxcommon.color/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!?
*common.apparel_class/MatMul/ReadVariableOpReadVariableOp3common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
common.apparel_class/MatMulMatMuldense_main_last/Tanh:y:02common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp4common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
common.apparel_class/BiasAddBiasAdd%common.apparel_class/MatMul:product:03common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
common.apparel_class/SoftmaxSoftmax%common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

Identity_1Identitycommon.color/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp,^common.apparel_class/BiasAdd/ReadVariableOp+^common.apparel_class/MatMul/ReadVariableOp$^common.color/BiasAdd/ReadVariableOp#^common.color/MatMul/ReadVariableOp#^conv_main_1/BiasAdd/ReadVariableOp"^conv_main_1/Conv2D/ReadVariableOp#^conv_main_2/BiasAdd/ReadVariableOp"^conv_main_2/Conv2D/ReadVariableOp#^conv_main_3/BiasAdd/ReadVariableOp"^conv_main_3/Conv2D/ReadVariableOp$^dense_main_1/BiasAdd/ReadVariableOp#^dense_main_1/MatMul/ReadVariableOp$^dense_main_2/BiasAdd/ReadVariableOp#^dense_main_2/MatMul/ReadVariableOp'^dense_main_last/BiasAdd/ReadVariableOp&^dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2Z
+common.apparel_class/BiasAdd/ReadVariableOp+common.apparel_class/BiasAdd/ReadVariableOp2X
*common.apparel_class/MatMul/ReadVariableOp*common.apparel_class/MatMul/ReadVariableOp2J
#common.color/BiasAdd/ReadVariableOp#common.color/BiasAdd/ReadVariableOp2H
"common.color/MatMul/ReadVariableOp"common.color/MatMul/ReadVariableOp2H
"conv_main_1/BiasAdd/ReadVariableOp"conv_main_1/BiasAdd/ReadVariableOp2F
!conv_main_1/Conv2D/ReadVariableOp!conv_main_1/Conv2D/ReadVariableOp2H
"conv_main_2/BiasAdd/ReadVariableOp"conv_main_2/BiasAdd/ReadVariableOp2F
!conv_main_2/Conv2D/ReadVariableOp!conv_main_2/Conv2D/ReadVariableOp2H
"conv_main_3/BiasAdd/ReadVariableOp"conv_main_3/BiasAdd/ReadVariableOp2F
!conv_main_3/Conv2D/ReadVariableOp!conv_main_3/Conv2D/ReadVariableOp2J
#dense_main_1/BiasAdd/ReadVariableOp#dense_main_1/BiasAdd/ReadVariableOp2H
"dense_main_1/MatMul/ReadVariableOp"dense_main_1/MatMul/ReadVariableOp2J
#dense_main_2/BiasAdd/ReadVariableOp#dense_main_2/BiasAdd/ReadVariableOp2H
"dense_main_2/MatMul/ReadVariableOp"dense_main_2/MatMul/ReadVariableOp2P
&dense_main_last/BiasAdd/ReadVariableOp&dense_main_last/BiasAdd/ReadVariableOp2N
%dense_main_last/MatMul/ReadVariableOp%dense_main_last/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
J__inference_dense_main_last_layer_call_and_return_conditional_losses_44479

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv_main_2_layer_call_fn_44340

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_main_layer_call_fn_44402

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43734p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
e
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_44361

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?:
?
B__inference_model_1_layer_call_and_return_conditional_losses_43627

inputs+
conv_main_1_43484:
conv_main_1_43486:+
conv_main_2_43502:
conv_main_2_43504:+
conv_main_3_43520: 
conv_main_3_43522: &
dense_main_1_43552:
?@?!
dense_main_1_43554:	?&
dense_main_2_43569:
??!
dense_main_2_43571:	?)
dense_main_last_43586:
??$
dense_main_last_43588:	?%
common_color_43603:	?! 
common_color_43605:!-
common_apparel_class_43620:	?(
common_apparel_class_43622:
identity

identity_1??,common.apparel_class/StatefulPartitionedCall?$common.color/StatefulPartitionedCall?#conv_main_1/StatefulPartitionedCall?#conv_main_2/StatefulPartitionedCall?#conv_main_3/StatefulPartitionedCall?$dense_main_1/StatefulPartitionedCall?$dense_main_2/StatefulPartitionedCall?'dense_main_last/StatefulPartitionedCall?
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_main_1_43484conv_main_1_43486*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483?
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450?
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_43502conv_main_2_43504*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501?
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462?
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_43520conv_main_3_43522*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519?
flatten_1/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531?
dropout_main/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43538?
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall%dropout_main/PartitionedCall:output:0dense_main_1_43552dense_main_1_43554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551?
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_43569dense_main_2_43571*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568?
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_43586dense_main_last_43588*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585?
$common.color/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_color_43603common_color_43605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_common.color_layer_call_and_return_conditional_losses_43602?
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_43620common_apparel_class_43622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619?
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-common.color/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall%^common.color/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2L
$common.color/StatefulPartitionedCall$common.color/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_main_layer_call_and_return_conditional_losses_44407

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????@\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?]
?
B__inference_model_1_layer_call_and_return_conditional_losses_44301

inputsD
*conv_main_1_conv2d_readvariableop_resource:9
+conv_main_1_biasadd_readvariableop_resource:D
*conv_main_2_conv2d_readvariableop_resource:9
+conv_main_2_biasadd_readvariableop_resource:D
*conv_main_3_conv2d_readvariableop_resource: 9
+conv_main_3_biasadd_readvariableop_resource: ?
+dense_main_1_matmul_readvariableop_resource:
?@?;
,dense_main_1_biasadd_readvariableop_resource:	??
+dense_main_2_matmul_readvariableop_resource:
??;
,dense_main_2_biasadd_readvariableop_resource:	?B
.dense_main_last_matmul_readvariableop_resource:
??>
/dense_main_last_biasadd_readvariableop_resource:	?>
+common_color_matmul_readvariableop_resource:	?!:
,common_color_biasadd_readvariableop_resource:!F
3common_apparel_class_matmul_readvariableop_resource:	?B
4common_apparel_class_biasadd_readvariableop_resource:
identity

identity_1??+common.apparel_class/BiasAdd/ReadVariableOp?*common.apparel_class/MatMul/ReadVariableOp?#common.color/BiasAdd/ReadVariableOp?"common.color/MatMul/ReadVariableOp?"conv_main_1/BiasAdd/ReadVariableOp?!conv_main_1/Conv2D/ReadVariableOp?"conv_main_2/BiasAdd/ReadVariableOp?!conv_main_2/Conv2D/ReadVariableOp?"conv_main_3/BiasAdd/ReadVariableOp?!conv_main_3/Conv2D/ReadVariableOp?#dense_main_1/BiasAdd/ReadVariableOp?"dense_main_1/MatMul/ReadVariableOp?#dense_main_2/BiasAdd/ReadVariableOp?"dense_main_2/MatMul/ReadVariableOp?&dense_main_last/BiasAdd/ReadVariableOp?%dense_main_last/MatMul/ReadVariableOp?
!conv_main_1/Conv2D/ReadVariableOpReadVariableOp*conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_main_1/Conv2DConv2Dinputs)conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
"conv_main_1/BiasAdd/ReadVariableOpReadVariableOp+conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_main_1/BiasAddBiasAddconv_main_1/Conv2D:output:0*conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????r
conv_main_1/ReluReluconv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
maxpool_main_1/MaxPoolMaxPoolconv_main_1/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
?
!conv_main_2/Conv2D/ReadVariableOpReadVariableOp*conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_main_2/Conv2DConv2Dmaxpool_main_1/MaxPool:output:0)conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
"conv_main_2/BiasAdd/ReadVariableOpReadVariableOp+conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_main_2/BiasAddBiasAddconv_main_2/Conv2D:output:0*conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@p
conv_main_2/ReluReluconv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
maxpool_main_2/MaxPoolMaxPoolconv_main_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
!conv_main_3/Conv2D/ReadVariableOpReadVariableOp*conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv_main_3/Conv2DConv2Dmaxpool_main_2/MaxPool:output:0)conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
"conv_main_3/BiasAdd/ReadVariableOpReadVariableOp+conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv_main_3/BiasAddBiasAddconv_main_3/Conv2D:output:0*conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? p
conv_main_3/ReluReluconv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten_1/ReshapeReshapeconv_main_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????@_
dropout_main/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_main/dropout/MulMulflatten_1/Reshape:output:0#dropout_main/dropout/Const:output:0*
T0*(
_output_shapes
:??????????@d
dropout_main/dropout/ShapeShapeflatten_1/Reshape:output:0*
T0*
_output_shapes
:?
1dropout_main/dropout/random_uniform/RandomUniformRandomUniform#dropout_main/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????@*
dtype0h
#dropout_main/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
!dropout_main/dropout/GreaterEqualGreaterEqual:dropout_main/dropout/random_uniform/RandomUniform:output:0,dropout_main/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????@?
dropout_main/dropout/CastCast%dropout_main/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????@?
dropout_main/dropout/Mul_1Muldropout_main/dropout/Mul:z:0dropout_main/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????@?
"dense_main_1/MatMul/ReadVariableOpReadVariableOp+dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
?@?*
dtype0?
dense_main_1/MatMulMatMuldropout_main/dropout/Mul_1:z:0*dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#dense_main_1/BiasAdd/ReadVariableOpReadVariableOp,dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_1/BiasAddBiasAdddense_main_1/MatMul:product:0+dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
dense_main_1/ReluReludense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"dense_main_2/MatMul/ReadVariableOpReadVariableOp+dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_main_2/MatMulMatMuldense_main_1/Relu:activations:0*dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#dense_main_2/BiasAdd/ReadVariableOpReadVariableOp,dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_2/BiasAddBiasAdddense_main_2/MatMul:product:0+dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
dense_main_2/ReluReludense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
%dense_main_last/MatMul/ReadVariableOpReadVariableOp.dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_main_last/MatMulMatMuldense_main_2/Relu:activations:0-dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&dense_main_last/BiasAdd/ReadVariableOpReadVariableOp/dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_main_last/BiasAddBiasAdd dense_main_last/MatMul:product:0.dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
dense_main_last/TanhTanh dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"common.color/MatMul/ReadVariableOpReadVariableOp+common_color_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype0?
common.color/MatMulMatMuldense_main_last/Tanh:y:0*common.color/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!?
#common.color/BiasAdd/ReadVariableOpReadVariableOp,common_color_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype0?
common.color/BiasAddBiasAddcommon.color/MatMul:product:0+common.color/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!p
common.color/SoftmaxSoftmaxcommon.color/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!?
*common.apparel_class/MatMul/ReadVariableOpReadVariableOp3common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
common.apparel_class/MatMulMatMuldense_main_last/Tanh:y:02common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp4common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
common.apparel_class/BiasAddBiasAdd%common.apparel_class/MatMul:product:03common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
common.apparel_class/SoftmaxSoftmax%common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????o

Identity_1Identitycommon.color/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp,^common.apparel_class/BiasAdd/ReadVariableOp+^common.apparel_class/MatMul/ReadVariableOp$^common.color/BiasAdd/ReadVariableOp#^common.color/MatMul/ReadVariableOp#^conv_main_1/BiasAdd/ReadVariableOp"^conv_main_1/Conv2D/ReadVariableOp#^conv_main_2/BiasAdd/ReadVariableOp"^conv_main_2/Conv2D/ReadVariableOp#^conv_main_3/BiasAdd/ReadVariableOp"^conv_main_3/Conv2D/ReadVariableOp$^dense_main_1/BiasAdd/ReadVariableOp#^dense_main_1/MatMul/ReadVariableOp$^dense_main_2/BiasAdd/ReadVariableOp#^dense_main_2/MatMul/ReadVariableOp'^dense_main_last/BiasAdd/ReadVariableOp&^dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2Z
+common.apparel_class/BiasAdd/ReadVariableOp+common.apparel_class/BiasAdd/ReadVariableOp2X
*common.apparel_class/MatMul/ReadVariableOp*common.apparel_class/MatMul/ReadVariableOp2J
#common.color/BiasAdd/ReadVariableOp#common.color/BiasAdd/ReadVariableOp2H
"common.color/MatMul/ReadVariableOp"common.color/MatMul/ReadVariableOp2H
"conv_main_1/BiasAdd/ReadVariableOp"conv_main_1/BiasAdd/ReadVariableOp2F
!conv_main_1/Conv2D/ReadVariableOp!conv_main_1/Conv2D/ReadVariableOp2H
"conv_main_2/BiasAdd/ReadVariableOp"conv_main_2/BiasAdd/ReadVariableOp2F
!conv_main_2/Conv2D/ReadVariableOp!conv_main_2/Conv2D/ReadVariableOp2H
"conv_main_3/BiasAdd/ReadVariableOp"conv_main_3/BiasAdd/ReadVariableOp2F
!conv_main_3/Conv2D/ReadVariableOp!conv_main_3/Conv2D/ReadVariableOp2J
#dense_main_1/BiasAdd/ReadVariableOp#dense_main_1/BiasAdd/ReadVariableOp2H
"dense_main_1/MatMul/ReadVariableOp"dense_main_1/MatMul/ReadVariableOp2J
#dense_main_2/BiasAdd/ReadVariableOp#dense_main_2/BiasAdd/ReadVariableOp2H
"dense_main_2/MatMul/ReadVariableOp"dense_main_2/MatMul/ReadVariableOp2P
&dense_main_last/BiasAdd/ReadVariableOp&dense_main_last/BiasAdd/ReadVariableOp2N
%dense_main_last/MatMul/ReadVariableOp%dense_main_last/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_conv_main_3_layer_call_and_return_conditional_losses_44381

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_main_1_layer_call_fn_44428

inputs
unknown:
?@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
,__inference_common.color_layer_call_fn_44508

inputs
unknown:	?!
	unknown_0:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_common.color_layer_call_and_return_conditional_losses_43602o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
+__inference_conv_main_3_layer_call_fn_44370

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_43664	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
?@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?!

unknown_12:!

unknown_13:	?

unknown_14:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_43627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
,__inference_dense_main_2_layer_call_fn_44448

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_44162

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
?@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?!

unknown_12:!

unknown_13:	?

unknown_14:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_43863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_conv_main_2_layer_call_and_return_conditional_losses_44351

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
4__inference_common.apparel_class_layer_call_fn_44488

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?'
!__inference__traced_restore_44931
file_prefix=
#assignvariableop_conv_main_1_kernel:1
#assignvariableop_1_conv_main_1_bias:?
%assignvariableop_2_conv_main_2_kernel:1
#assignvariableop_3_conv_main_2_bias:?
%assignvariableop_4_conv_main_3_kernel: 1
#assignvariableop_5_conv_main_3_bias: :
&assignvariableop_6_dense_main_1_kernel:
?@?3
$assignvariableop_7_dense_main_1_bias:	?:
&assignvariableop_8_dense_main_2_kernel:
??3
$assignvariableop_9_dense_main_2_bias:	?>
*assignvariableop_10_dense_main_last_kernel:
??7
(assignvariableop_11_dense_main_last_bias:	?B
/assignvariableop_12_common_apparel_class_kernel:	?;
-assignvariableop_13_common_apparel_class_bias::
'assignvariableop_14_common_color_kernel:	?!3
%assignvariableop_15_common_color_bias:!'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_4: %
assignvariableop_22_count_4: %
assignvariableop_23_total_3: %
assignvariableop_24_count_3: %
assignvariableop_25_total_2: %
assignvariableop_26_count_2: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: #
assignvariableop_29_total: #
assignvariableop_30_count: G
-assignvariableop_31_adam_conv_main_1_kernel_m:9
+assignvariableop_32_adam_conv_main_1_bias_m:G
-assignvariableop_33_adam_conv_main_2_kernel_m:9
+assignvariableop_34_adam_conv_main_2_bias_m:G
-assignvariableop_35_adam_conv_main_3_kernel_m: 9
+assignvariableop_36_adam_conv_main_3_bias_m: B
.assignvariableop_37_adam_dense_main_1_kernel_m:
?@?;
,assignvariableop_38_adam_dense_main_1_bias_m:	?B
.assignvariableop_39_adam_dense_main_2_kernel_m:
??;
,assignvariableop_40_adam_dense_main_2_bias_m:	?E
1assignvariableop_41_adam_dense_main_last_kernel_m:
??>
/assignvariableop_42_adam_dense_main_last_bias_m:	?I
6assignvariableop_43_adam_common_apparel_class_kernel_m:	?B
4assignvariableop_44_adam_common_apparel_class_bias_m:A
.assignvariableop_45_adam_common_color_kernel_m:	?!:
,assignvariableop_46_adam_common_color_bias_m:!G
-assignvariableop_47_adam_conv_main_1_kernel_v:9
+assignvariableop_48_adam_conv_main_1_bias_v:G
-assignvariableop_49_adam_conv_main_2_kernel_v:9
+assignvariableop_50_adam_conv_main_2_bias_v:G
-assignvariableop_51_adam_conv_main_3_kernel_v: 9
+assignvariableop_52_adam_conv_main_3_bias_v: B
.assignvariableop_53_adam_dense_main_1_kernel_v:
?@?;
,assignvariableop_54_adam_dense_main_1_bias_v:	?B
.assignvariableop_55_adam_dense_main_2_kernel_v:
??;
,assignvariableop_56_adam_dense_main_2_bias_v:	?E
1assignvariableop_57_adam_dense_main_last_kernel_v:
??>
/assignvariableop_58_adam_dense_main_last_bias_v:	?I
6assignvariableop_59_adam_common_apparel_class_kernel_v:	?B
4assignvariableop_60_adam_common_apparel_class_bias_v:A
.assignvariableop_61_adam_common_color_kernel_v:	?!:
,assignvariableop_62_adam_common_color_bias_v:!
identity_64??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?"
value?!B?!@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp#assignvariableop_conv_main_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv_main_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv_main_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv_main_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv_main_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv_main_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_dense_main_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_dense_main_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_dense_main_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_dense_main_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp*assignvariableop_10_dense_main_last_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp(assignvariableop_11_dense_main_last_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_common_apparel_class_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_common_apparel_class_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_common_color_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_common_color_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_4Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_4Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_3Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_conv_main_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_conv_main_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_conv_main_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_conv_main_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_conv_main_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_conv_main_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_dense_main_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_dense_main_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_dense_main_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_dense_main_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_dense_main_last_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_dense_main_last_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_common_apparel_class_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_common_apparel_class_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_common_color_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_common_color_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_conv_main_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_conv_main_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_conv_main_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_conv_main_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_conv_main_3_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_conv_main_3_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp.assignvariableop_53_adam_dense_main_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp,assignvariableop_54_adam_dense_main_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp.assignvariableop_55_adam_dense_main_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_dense_main_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp1assignvariableop_57_adam_dense_main_last_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp/assignvariableop_58_adam_dense_main_last_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_common_apparel_class_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adam_common_apparel_class_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_common_color_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_common_color_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
e
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_main_layer_call_fn_44397

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43538a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_43939	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
?@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?!

unknown_12:!

unknown_13:	?

unknown_14:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_43863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
#__inference_signature_wrapper_44084	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
?@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?!

unknown_12:!

unknown_13:	?

unknown_14:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_43441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
+__inference_conv_main_1_layer_call_fn_44310

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
f
G__inference_dropout_main_layer_call_and_return_conditional_losses_43734

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????@p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????@j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????@Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_44392

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?|
?
__inference__traced_save_44732
file_prefix1
-savev2_conv_main_1_kernel_read_readvariableop/
+savev2_conv_main_1_bias_read_readvariableop1
-savev2_conv_main_2_kernel_read_readvariableop/
+savev2_conv_main_2_bias_read_readvariableop1
-savev2_conv_main_3_kernel_read_readvariableop/
+savev2_conv_main_3_bias_read_readvariableop2
.savev2_dense_main_1_kernel_read_readvariableop0
,savev2_dense_main_1_bias_read_readvariableop2
.savev2_dense_main_2_kernel_read_readvariableop0
,savev2_dense_main_2_bias_read_readvariableop5
1savev2_dense_main_last_kernel_read_readvariableop3
/savev2_dense_main_last_bias_read_readvariableop:
6savev2_common_apparel_class_kernel_read_readvariableop8
4savev2_common_apparel_class_bias_read_readvariableop2
.savev2_common_color_kernel_read_readvariableop0
,savev2_common_color_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_conv_main_1_kernel_m_read_readvariableop6
2savev2_adam_conv_main_1_bias_m_read_readvariableop8
4savev2_adam_conv_main_2_kernel_m_read_readvariableop6
2savev2_adam_conv_main_2_bias_m_read_readvariableop8
4savev2_adam_conv_main_3_kernel_m_read_readvariableop6
2savev2_adam_conv_main_3_bias_m_read_readvariableop9
5savev2_adam_dense_main_1_kernel_m_read_readvariableop7
3savev2_adam_dense_main_1_bias_m_read_readvariableop9
5savev2_adam_dense_main_2_kernel_m_read_readvariableop7
3savev2_adam_dense_main_2_bias_m_read_readvariableop<
8savev2_adam_dense_main_last_kernel_m_read_readvariableop:
6savev2_adam_dense_main_last_bias_m_read_readvariableopA
=savev2_adam_common_apparel_class_kernel_m_read_readvariableop?
;savev2_adam_common_apparel_class_bias_m_read_readvariableop9
5savev2_adam_common_color_kernel_m_read_readvariableop7
3savev2_adam_common_color_bias_m_read_readvariableop8
4savev2_adam_conv_main_1_kernel_v_read_readvariableop6
2savev2_adam_conv_main_1_bias_v_read_readvariableop8
4savev2_adam_conv_main_2_kernel_v_read_readvariableop6
2savev2_adam_conv_main_2_bias_v_read_readvariableop8
4savev2_adam_conv_main_3_kernel_v_read_readvariableop6
2savev2_adam_conv_main_3_bias_v_read_readvariableop9
5savev2_adam_dense_main_1_kernel_v_read_readvariableop7
3savev2_adam_dense_main_1_bias_v_read_readvariableop9
5savev2_adam_dense_main_2_kernel_v_read_readvariableop7
3savev2_adam_dense_main_2_bias_v_read_readvariableop<
8savev2_adam_dense_main_last_kernel_v_read_readvariableop:
6savev2_adam_dense_main_last_bias_v_read_readvariableopA
=savev2_adam_common_apparel_class_kernel_v_read_readvariableop?
;savev2_adam_common_apparel_class_bias_v_read_readvariableop9
5savev2_adam_common_color_kernel_v_read_readvariableop7
3savev2_adam_common_color_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?"
value?!B?!@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*?
value?B?@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_conv_main_1_kernel_read_readvariableop+savev2_conv_main_1_bias_read_readvariableop-savev2_conv_main_2_kernel_read_readvariableop+savev2_conv_main_2_bias_read_readvariableop-savev2_conv_main_3_kernel_read_readvariableop+savev2_conv_main_3_bias_read_readvariableop.savev2_dense_main_1_kernel_read_readvariableop,savev2_dense_main_1_bias_read_readvariableop.savev2_dense_main_2_kernel_read_readvariableop,savev2_dense_main_2_bias_read_readvariableop1savev2_dense_main_last_kernel_read_readvariableop/savev2_dense_main_last_bias_read_readvariableop6savev2_common_apparel_class_kernel_read_readvariableop4savev2_common_apparel_class_bias_read_readvariableop.savev2_common_color_kernel_read_readvariableop,savev2_common_color_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_conv_main_1_kernel_m_read_readvariableop2savev2_adam_conv_main_1_bias_m_read_readvariableop4savev2_adam_conv_main_2_kernel_m_read_readvariableop2savev2_adam_conv_main_2_bias_m_read_readvariableop4savev2_adam_conv_main_3_kernel_m_read_readvariableop2savev2_adam_conv_main_3_bias_m_read_readvariableop5savev2_adam_dense_main_1_kernel_m_read_readvariableop3savev2_adam_dense_main_1_bias_m_read_readvariableop5savev2_adam_dense_main_2_kernel_m_read_readvariableop3savev2_adam_dense_main_2_bias_m_read_readvariableop8savev2_adam_dense_main_last_kernel_m_read_readvariableop6savev2_adam_dense_main_last_bias_m_read_readvariableop=savev2_adam_common_apparel_class_kernel_m_read_readvariableop;savev2_adam_common_apparel_class_bias_m_read_readvariableop5savev2_adam_common_color_kernel_m_read_readvariableop3savev2_adam_common_color_bias_m_read_readvariableop4savev2_adam_conv_main_1_kernel_v_read_readvariableop2savev2_adam_conv_main_1_bias_v_read_readvariableop4savev2_adam_conv_main_2_kernel_v_read_readvariableop2savev2_adam_conv_main_2_bias_v_read_readvariableop4savev2_adam_conv_main_3_kernel_v_read_readvariableop2savev2_adam_conv_main_3_bias_v_read_readvariableop5savev2_adam_dense_main_1_kernel_v_read_readvariableop3savev2_adam_dense_main_1_bias_v_read_readvariableop5savev2_adam_dense_main_2_kernel_v_read_readvariableop3savev2_adam_dense_main_2_bias_v_read_readvariableop8savev2_adam_dense_main_last_kernel_v_read_readvariableop6savev2_adam_dense_main_last_bias_v_read_readvariableop=savev2_adam_common_apparel_class_kernel_v_read_readvariableop;savev2_adam_common_apparel_class_bias_v_read_readvariableop5savev2_adam_common_color_kernel_v_read_readvariableop3savev2_adam_common_color_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : :
?@?:?:
??:?:
??:?:	?::	?!:!: : : : : : : : : : : : : : : ::::: : :
?@?:?:
??:?:
??:?:	?::	?!:!::::: : :
?@?:?:
??:?:
??:?:	?::	?!:!: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?@?:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?!: 

_output_shapes
:!:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
: : %

_output_shapes
: :&&"
 
_output_shapes
:
?@?:!'

_output_shapes	
:?:&("
 
_output_shapes
:
??:!)

_output_shapes	
:?:&*"
 
_output_shapes
:
??:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::%.!

_output_shapes
:	?!: /

_output_shapes
:!:,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: :&6"
 
_output_shapes
:
?@?:!7

_output_shapes	
:?:&8"
 
_output_shapes
:
??:!9

_output_shapes	
:?:&:"
 
_output_shapes
:
??:!;

_output_shapes	
:?:%<!

_output_shapes
:	?: =

_output_shapes
::%>!

_output_shapes
:	?!: ?

_output_shapes
:!:@

_output_shapes
: 
?	
f
G__inference_dropout_main_layer_call_and_return_conditional_losses_44419

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????@p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????@j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????@Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
/__inference_dense_main_last_layer_call_fn_44468

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv_main_1_layer_call_and_return_conditional_losses_44321

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_maxpool_main_2_layer_call_fn_44356

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_main_layer_call_and_return_conditional_losses_43538

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????@\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?`
?
 __inference__wrapped_model_43441	
inputL
2model_1_conv_main_1_conv2d_readvariableop_resource:A
3model_1_conv_main_1_biasadd_readvariableop_resource:L
2model_1_conv_main_2_conv2d_readvariableop_resource:A
3model_1_conv_main_2_biasadd_readvariableop_resource:L
2model_1_conv_main_3_conv2d_readvariableop_resource: A
3model_1_conv_main_3_biasadd_readvariableop_resource: G
3model_1_dense_main_1_matmul_readvariableop_resource:
?@?C
4model_1_dense_main_1_biasadd_readvariableop_resource:	?G
3model_1_dense_main_2_matmul_readvariableop_resource:
??C
4model_1_dense_main_2_biasadd_readvariableop_resource:	?J
6model_1_dense_main_last_matmul_readvariableop_resource:
??F
7model_1_dense_main_last_biasadd_readvariableop_resource:	?F
3model_1_common_color_matmul_readvariableop_resource:	?!B
4model_1_common_color_biasadd_readvariableop_resource:!N
;model_1_common_apparel_class_matmul_readvariableop_resource:	?J
<model_1_common_apparel_class_biasadd_readvariableop_resource:
identity

identity_1??3model_1/common.apparel_class/BiasAdd/ReadVariableOp?2model_1/common.apparel_class/MatMul/ReadVariableOp?+model_1/common.color/BiasAdd/ReadVariableOp?*model_1/common.color/MatMul/ReadVariableOp?*model_1/conv_main_1/BiasAdd/ReadVariableOp?)model_1/conv_main_1/Conv2D/ReadVariableOp?*model_1/conv_main_2/BiasAdd/ReadVariableOp?)model_1/conv_main_2/Conv2D/ReadVariableOp?*model_1/conv_main_3/BiasAdd/ReadVariableOp?)model_1/conv_main_3/Conv2D/ReadVariableOp?+model_1/dense_main_1/BiasAdd/ReadVariableOp?*model_1/dense_main_1/MatMul/ReadVariableOp?+model_1/dense_main_2/BiasAdd/ReadVariableOp?*model_1/dense_main_2/MatMul/ReadVariableOp?.model_1/dense_main_last/BiasAdd/ReadVariableOp?-model_1/dense_main_last/MatMul/ReadVariableOp?
)model_1/conv_main_1/Conv2D/ReadVariableOpReadVariableOp2model_1_conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_1/conv_main_1/Conv2DConv2Dinput1model_1/conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*model_1/conv_main_1/BiasAdd/ReadVariableOpReadVariableOp3model_1_conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv_main_1/BiasAddBiasAdd#model_1/conv_main_1/Conv2D:output:02model_1/conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
model_1/conv_main_1/ReluRelu$model_1/conv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
model_1/maxpool_main_1/MaxPoolMaxPool&model_1/conv_main_1/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
?
)model_1/conv_main_2/Conv2D/ReadVariableOpReadVariableOp2model_1_conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_1/conv_main_2/Conv2DConv2D'model_1/maxpool_main_1/MaxPool:output:01model_1/conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
*model_1/conv_main_2/BiasAdd/ReadVariableOpReadVariableOp3model_1_conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv_main_2/BiasAddBiasAdd#model_1/conv_main_2/Conv2D:output:02model_1/conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
model_1/conv_main_2/ReluRelu$model_1/conv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
model_1/maxpool_main_2/MaxPoolMaxPool&model_1/conv_main_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
)model_1/conv_main_3/Conv2D/ReadVariableOpReadVariableOp2model_1_conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model_1/conv_main_3/Conv2DConv2D'model_1/maxpool_main_2/MaxPool:output:01model_1/conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*model_1/conv_main_3/BiasAdd/ReadVariableOpReadVariableOp3model_1_conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_1/conv_main_3/BiasAddBiasAdd#model_1/conv_main_3/Conv2D:output:02model_1/conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
model_1/conv_main_3/ReluRelu$model_1/conv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? h
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
model_1/flatten_1/ReshapeReshape&model_1/conv_main_3/Relu:activations:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????@?
model_1/dropout_main/IdentityIdentity"model_1/flatten_1/Reshape:output:0*
T0*(
_output_shapes
:??????????@?
*model_1/dense_main_1/MatMul/ReadVariableOpReadVariableOp3model_1_dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
?@?*
dtype0?
model_1/dense_main_1/MatMulMatMul&model_1/dropout_main/Identity:output:02model_1/dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+model_1/dense_main_1/BiasAdd/ReadVariableOpReadVariableOp4model_1_dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_main_1/BiasAddBiasAdd%model_1/dense_main_1/MatMul:product:03model_1/dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
model_1/dense_main_1/ReluRelu%model_1/dense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*model_1/dense_main_2/MatMul/ReadVariableOpReadVariableOp3model_1_dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_1/dense_main_2/MatMulMatMul'model_1/dense_main_1/Relu:activations:02model_1/dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+model_1/dense_main_2/BiasAdd/ReadVariableOpReadVariableOp4model_1_dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_main_2/BiasAddBiasAdd%model_1/dense_main_2/MatMul:product:03model_1/dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
model_1/dense_main_2/ReluRelu%model_1/dense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-model_1/dense_main_last/MatMul/ReadVariableOpReadVariableOp6model_1_dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_1/dense_main_last/MatMulMatMul'model_1/dense_main_2/Relu:activations:05model_1/dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.model_1/dense_main_last/BiasAdd/ReadVariableOpReadVariableOp7model_1_dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_main_last/BiasAddBiasAdd(model_1/dense_main_last/MatMul:product:06model_1/dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
model_1/dense_main_last/TanhTanh(model_1/dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*model_1/common.color/MatMul/ReadVariableOpReadVariableOp3model_1_common_color_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype0?
model_1/common.color/MatMulMatMul model_1/dense_main_last/Tanh:y:02model_1/common.color/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!?
+model_1/common.color/BiasAdd/ReadVariableOpReadVariableOp4model_1_common_color_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype0?
model_1/common.color/BiasAddBiasAdd%model_1/common.color/MatMul:product:03model_1/common.color/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!?
model_1/common.color/SoftmaxSoftmax%model_1/common.color/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!?
2model_1/common.apparel_class/MatMul/ReadVariableOpReadVariableOp;model_1_common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
#model_1/common.apparel_class/MatMulMatMul model_1/dense_main_last/Tanh:y:0:model_1/common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3model_1/common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp<model_1_common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$model_1/common.apparel_class/BiasAddBiasAdd-model_1/common.apparel_class/MatMul:product:0;model_1/common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model_1/common.apparel_class/SoftmaxSoftmax-model_1/common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
IdentityIdentity.model_1/common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_1Identity&model_1/common.color/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp4^model_1/common.apparel_class/BiasAdd/ReadVariableOp3^model_1/common.apparel_class/MatMul/ReadVariableOp,^model_1/common.color/BiasAdd/ReadVariableOp+^model_1/common.color/MatMul/ReadVariableOp+^model_1/conv_main_1/BiasAdd/ReadVariableOp*^model_1/conv_main_1/Conv2D/ReadVariableOp+^model_1/conv_main_2/BiasAdd/ReadVariableOp*^model_1/conv_main_2/Conv2D/ReadVariableOp+^model_1/conv_main_3/BiasAdd/ReadVariableOp*^model_1/conv_main_3/Conv2D/ReadVariableOp,^model_1/dense_main_1/BiasAdd/ReadVariableOp+^model_1/dense_main_1/MatMul/ReadVariableOp,^model_1/dense_main_2/BiasAdd/ReadVariableOp+^model_1/dense_main_2/MatMul/ReadVariableOp/^model_1/dense_main_last/BiasAdd/ReadVariableOp.^model_1/dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2j
3model_1/common.apparel_class/BiasAdd/ReadVariableOp3model_1/common.apparel_class/BiasAdd/ReadVariableOp2h
2model_1/common.apparel_class/MatMul/ReadVariableOp2model_1/common.apparel_class/MatMul/ReadVariableOp2Z
+model_1/common.color/BiasAdd/ReadVariableOp+model_1/common.color/BiasAdd/ReadVariableOp2X
*model_1/common.color/MatMul/ReadVariableOp*model_1/common.color/MatMul/ReadVariableOp2X
*model_1/conv_main_1/BiasAdd/ReadVariableOp*model_1/conv_main_1/BiasAdd/ReadVariableOp2V
)model_1/conv_main_1/Conv2D/ReadVariableOp)model_1/conv_main_1/Conv2D/ReadVariableOp2X
*model_1/conv_main_2/BiasAdd/ReadVariableOp*model_1/conv_main_2/BiasAdd/ReadVariableOp2V
)model_1/conv_main_2/Conv2D/ReadVariableOp)model_1/conv_main_2/Conv2D/ReadVariableOp2X
*model_1/conv_main_3/BiasAdd/ReadVariableOp*model_1/conv_main_3/BiasAdd/ReadVariableOp2V
)model_1/conv_main_3/Conv2D/ReadVariableOp)model_1/conv_main_3/Conv2D/ReadVariableOp2Z
+model_1/dense_main_1/BiasAdd/ReadVariableOp+model_1/dense_main_1/BiasAdd/ReadVariableOp2X
*model_1/dense_main_1/MatMul/ReadVariableOp*model_1/dense_main_1/MatMul/ReadVariableOp2Z
+model_1/dense_main_2/BiasAdd/ReadVariableOp+model_1/dense_main_2/BiasAdd/ReadVariableOp2X
*model_1/dense_main_2/MatMul/ReadVariableOp*model_1/dense_main_2/MatMul/ReadVariableOp2`
.model_1/dense_main_last/BiasAdd/ReadVariableOp.model_1/dense_main_last/BiasAdd/ReadVariableOp2^
-model_1/dense_main_last/MatMul/ReadVariableOp-model_1/dense_main_last/MatMul/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?

?
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551

inputs2
matmul_readvariableop_resource:
?@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?<
?
B__inference_model_1_layer_call_and_return_conditional_losses_43863

inputs+
conv_main_1_43817:
conv_main_1_43819:+
conv_main_2_43823:
conv_main_2_43825:+
conv_main_3_43829: 
conv_main_3_43831: &
dense_main_1_43836:
?@?!
dense_main_1_43838:	?&
dense_main_2_43841:
??!
dense_main_2_43843:	?)
dense_main_last_43846:
??$
dense_main_last_43848:	?%
common_color_43851:	?! 
common_color_43853:!-
common_apparel_class_43856:	?(
common_apparel_class_43858:
identity

identity_1??,common.apparel_class/StatefulPartitionedCall?$common.color/StatefulPartitionedCall?#conv_main_1/StatefulPartitionedCall?#conv_main_2/StatefulPartitionedCall?#conv_main_3/StatefulPartitionedCall?$dense_main_1/StatefulPartitionedCall?$dense_main_2/StatefulPartitionedCall?'dense_main_last/StatefulPartitionedCall?$dropout_main/StatefulPartitionedCall?
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_main_1_43817conv_main_1_43819*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483?
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450?
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_43823conv_main_2_43825*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501?
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462?
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_43829conv_main_3_43831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519?
flatten_1/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531?
$dropout_main/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43734?
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall-dropout_main/StatefulPartitionedCall:output:0dense_main_1_43836dense_main_1_43838*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551?
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_43841dense_main_2_43843*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568?
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_43846dense_main_last_43848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585?
$common.color/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_color_43851common_color_43853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_common.color_layer_call_and_return_conditional_losses_43602?
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_43856common_apparel_class_43858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619?
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-common.color/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall%^common.color/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall%^dropout_main/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2L
$common.color/StatefulPartitionedCall$common.color/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall2L
$dropout_main/StatefulPartitionedCall$dropout_main/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
G__inference_dense_main_1_layer_call_and_return_conditional_losses_44439

inputs2
matmul_readvariableop_resource:
?@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

?
G__inference_common.color_layer_call_and_return_conditional_losses_43602

inputs1
matmul_readvariableop_resource:	?!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?!*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????!w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_44499

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_44331

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_maxpool_main_1_layer_call_fn_44326

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_common.color_layer_call_and_return_conditional_losses_44519

inputs1
matmul_readvariableop_resource:	?!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?!*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????!w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_dense_main_2_layer_call_and_return_conditional_losses_44459

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_44386

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
B__inference_model_1_layer_call_and_return_conditional_losses_44037	
input+
conv_main_1_43991:
conv_main_1_43993:+
conv_main_2_43997:
conv_main_2_43999:+
conv_main_3_44003: 
conv_main_3_44005: &
dense_main_1_44010:
?@?!
dense_main_1_44012:	?&
dense_main_2_44015:
??!
dense_main_2_44017:	?)
dense_main_last_44020:
??$
dense_main_last_44022:	?%
common_color_44025:	?! 
common_color_44027:!-
common_apparel_class_44030:	?(
common_apparel_class_44032:
identity

identity_1??,common.apparel_class/StatefulPartitionedCall?$common.color/StatefulPartitionedCall?#conv_main_1/StatefulPartitionedCall?#conv_main_2/StatefulPartitionedCall?#conv_main_3/StatefulPartitionedCall?$dense_main_1/StatefulPartitionedCall?$dense_main_2/StatefulPartitionedCall?'dense_main_last/StatefulPartitionedCall?$dropout_main/StatefulPartitionedCall?
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputconv_main_1_43991conv_main_1_43993*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_43483?
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_43450?
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_43997conv_main_2_43999*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_43501?
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_43462?
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_44003conv_main_3_44005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_43519?
flatten_1/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_43531?
$dropout_main/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_43734?
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall-dropout_main/StatefulPartitionedCall:output:0dense_main_1_44010dense_main_1_44012*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_43551?
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_44015dense_main_2_44017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_43568?
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_44020dense_main_last_44022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_43585?
$common.color/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_color_44025common_color_44027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_common.color_layer_call_and_return_conditional_losses_43602?
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_44030common_apparel_class_44032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_43619?
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-common.color/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????!?
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall%^common.color/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall%^dropout_main/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2L
$common.color/StatefulPartitionedCall$common.color/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall2L
$dropout_main/StatefulPartitionedCall$dropout_main/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
'__inference_model_1_layer_call_fn_44123

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
?@?
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?!

unknown_12:!

unknown_13:	?

unknown_14:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????!*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_43627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input8
serving_default_input:0???????????H
common.apparel_class0
StatefulPartitionedCall:0?????????@
common.color0
StatefulPartitionedCall:1?????????!tensorflow/serving/predict:Ǭ
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op"
_tf_keras_layer
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator"
_tf_keras_layer
?
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias"
_tf_keras_layer
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias"
_tf_keras_layer
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias"
_tf_keras_layer
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

jkernel
kbias"
_tf_keras_layer
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias"
_tf_keras_layer
?
0
1
-2
.3
<4
=5
R6
S7
Z8
[9
b10
c11
j12
k13
r14
s15"
trackable_list_wrapper
?
0
1
-2
.3
<4
=5
R6
S7
Z8
[9
b10
c11
j12
k13
r14
s15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
ytrace_0
ztrace_1
{trace_2
|trace_32?
'__inference_model_1_layer_call_fn_43664
'__inference_model_1_layer_call_fn_44123
'__inference_model_1_layer_call_fn_44162
'__inference_model_1_layer_call_fn_43939?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zytrace_0zztrace_1z{trace_2z|trace_3
?
}trace_0
~trace_1
trace_2
?trace_32?
B__inference_model_1_layer_call_and_return_conditional_losses_44228
B__inference_model_1_layer_call_and_return_conditional_losses_44301
B__inference_model_1_layer_call_and_return_conditional_losses_43988
B__inference_model_1_layer_call_and_return_conditional_losses_44037?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z}trace_0z~trace_1ztrace_2z?trace_3
?B?
 __inference__wrapped_model_43441input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?-m?.m?<m?=m?Rm?Sm?Zm?[m?bm?cm?jm?km?rm?sm?v?v?-v?.v?<v?=v?Rv?Sv?Zv?[v?bv?cv?jv?kv?rv?sv?"
	optimizer
 "
trackable_dict_wrapper
-
?serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv_main_1_layer_call_fn_44310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv_main_1_layer_call_and_return_conditional_losses_44321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*2conv_main_1/kernel
:2conv_main_1/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_maxpool_main_1_layer_call_fn_44326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_44331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv_main_2_layer_call_fn_44340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv_main_2_layer_call_and_return_conditional_losses_44351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*2conv_main_2/kernel
:2conv_main_2/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_maxpool_main_2_layer_call_fn_44356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_44361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv_main_3_layer_call_fn_44370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv_main_3_layer_call_and_return_conditional_losses_44381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:* 2conv_main_3/kernel
: 2conv_main_3/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_flatten_1_layer_call_fn_44386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_flatten_1_layer_call_and_return_conditional_losses_44392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
,__inference_dropout_main_layer_call_fn_44397
,__inference_dropout_main_layer_call_fn_44402?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
G__inference_dropout_main_layer_call_and_return_conditional_losses_44407
G__inference_dropout_main_layer_call_and_return_conditional_losses_44419?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_dense_main_1_layer_call_fn_44428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_dense_main_1_layer_call_and_return_conditional_losses_44439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
':%
?@?2dense_main_1/kernel
 :?2dense_main_1/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_dense_main_2_layer_call_fn_44448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_dense_main_2_layer_call_and_return_conditional_losses_44459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
':%
??2dense_main_2/kernel
 :?2dense_main_2/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_dense_main_last_layer_call_fn_44468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_dense_main_last_layer_call_and_return_conditional_losses_44479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(
??2dense_main_last/kernel
#:!?2dense_main_last/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
4__inference_common.apparel_class_layer_call_fn_44488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_44499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.:,	?2common.apparel_class/kernel
':%2common.apparel_class/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_common.color_layer_call_fn_44508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_common.color_layer_call_and_return_conditional_losses_44519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
&:$	?!2common.color/kernel
:!2common.color/bias
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_model_1_layer_call_fn_43664input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_44123inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_44162inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_43939input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_44228inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_44301inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_43988input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_44037input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_44084input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv_main_1_layer_call_fn_44310inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv_main_1_layer_call_and_return_conditional_losses_44321inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_maxpool_main_1_layer_call_fn_44326inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_44331inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv_main_2_layer_call_fn_44340inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv_main_2_layer_call_and_return_conditional_losses_44351inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_maxpool_main_2_layer_call_fn_44356inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_44361inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv_main_3_layer_call_fn_44370inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv_main_3_layer_call_and_return_conditional_losses_44381inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_flatten_1_layer_call_fn_44386inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_flatten_1_layer_call_and_return_conditional_losses_44392inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_dropout_main_layer_call_fn_44397inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_dropout_main_layer_call_fn_44402inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_dropout_main_layer_call_and_return_conditional_losses_44407inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_dropout_main_layer_call_and_return_conditional_losses_44419inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_dense_main_1_layer_call_fn_44428inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_dense_main_1_layer_call_and_return_conditional_losses_44439inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_dense_main_2_layer_call_fn_44448inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_dense_main_2_layer_call_and_return_conditional_losses_44459inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_dense_main_last_layer_call_fn_44468inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_dense_main_last_layer_call_and_return_conditional_losses_44479inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_common.apparel_class_layer_call_fn_44488inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_44499inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_common.color_layer_call_fn_44508inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_common.color_layer_call_and_return_conditional_losses_44519inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
1:/2Adam/conv_main_1/kernel/m
#:!2Adam/conv_main_1/bias/m
1:/2Adam/conv_main_2/kernel/m
#:!2Adam/conv_main_2/bias/m
1:/ 2Adam/conv_main_3/kernel/m
#:! 2Adam/conv_main_3/bias/m
,:*
?@?2Adam/dense_main_1/kernel/m
%:#?2Adam/dense_main_1/bias/m
,:*
??2Adam/dense_main_2/kernel/m
%:#?2Adam/dense_main_2/bias/m
/:-
??2Adam/dense_main_last/kernel/m
(:&?2Adam/dense_main_last/bias/m
3:1	?2"Adam/common.apparel_class/kernel/m
,:*2 Adam/common.apparel_class/bias/m
+:)	?!2Adam/common.color/kernel/m
$:"!2Adam/common.color/bias/m
1:/2Adam/conv_main_1/kernel/v
#:!2Adam/conv_main_1/bias/v
1:/2Adam/conv_main_2/kernel/v
#:!2Adam/conv_main_2/bias/v
1:/ 2Adam/conv_main_3/kernel/v
#:! 2Adam/conv_main_3/bias/v
,:*
?@?2Adam/dense_main_1/kernel/v
%:#?2Adam/dense_main_1/bias/v
,:*
??2Adam/dense_main_2/kernel/v
%:#?2Adam/dense_main_2/bias/v
/:-
??2Adam/dense_main_last/kernel/v
(:&?2Adam/dense_main_last/bias/v
3:1	?2"Adam/common.apparel_class/kernel/v
,:*2 Adam/common.apparel_class/bias/v
+:)	?!2Adam/common.color/kernel/v
$:"!2Adam/common.color/bias/v?
 __inference__wrapped_model_43441?-.<=RSZ[bcrsjk8?5
.?+
)?&
input???????????
? "???
F
common.apparel_class.?+
common.apparel_class?????????
6
common.color&?#
common.color?????????!?
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_44499]jk0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
4__inference_common.apparel_class_layer_call_fn_44488Pjk0?-
&?#
!?
inputs??????????
? "???????????
G__inference_common.color_layer_call_and_return_conditional_losses_44519]rs0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????!
? ?
,__inference_common.color_layer_call_fn_44508Prs0?-
&?#
!?
inputs??????????
? "??????????!?
F__inference_conv_main_1_layer_call_and_return_conditional_losses_44321p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv_main_1_layer_call_fn_44310c9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv_main_2_layer_call_and_return_conditional_losses_44351l-.7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
+__inference_conv_main_2_layer_call_fn_44340_-.7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
F__inference_conv_main_3_layer_call_and_return_conditional_losses_44381l<=7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
+__inference_conv_main_3_layer_call_fn_44370_<=7?4
-?*
(?%
inputs?????????
? " ?????????? ?
G__inference_dense_main_1_layer_call_and_return_conditional_losses_44439^RS0?-
&?#
!?
inputs??????????@
? "&?#
?
0??????????
? ?
,__inference_dense_main_1_layer_call_fn_44428QRS0?-
&?#
!?
inputs??????????@
? "????????????
G__inference_dense_main_2_layer_call_and_return_conditional_losses_44459^Z[0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_main_2_layer_call_fn_44448QZ[0?-
&?#
!?
inputs??????????
? "????????????
J__inference_dense_main_last_layer_call_and_return_conditional_losses_44479^bc0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_dense_main_last_layer_call_fn_44468Qbc0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dropout_main_layer_call_and_return_conditional_losses_44407^4?1
*?'
!?
inputs??????????@
p 
? "&?#
?
0??????????@
? ?
G__inference_dropout_main_layer_call_and_return_conditional_losses_44419^4?1
*?'
!?
inputs??????????@
p
? "&?#
?
0??????????@
? ?
,__inference_dropout_main_layer_call_fn_44397Q4?1
*?'
!?
inputs??????????@
p 
? "???????????@?
,__inference_dropout_main_layer_call_fn_44402Q4?1
*?'
!?
inputs??????????@
p
? "???????????@?
D__inference_flatten_1_layer_call_and_return_conditional_losses_44392a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????@
? ?
)__inference_flatten_1_layer_call_fn_44386T7?4
-?*
(?%
inputs????????? 
? "???????????@?
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_44331?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_maxpool_main_1_layer_call_fn_44326?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_44361?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_maxpool_main_2_layer_call_fn_44356?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_43988?-.<=RSZ[bcrsjk@?=
6?3
)?&
input???????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????!
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_44037?-.<=RSZ[bcrsjk@?=
6?3
)?&
input???????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????!
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_44228?-.<=RSZ[bcrsjkA?>
7?4
*?'
inputs???????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????!
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_44301?-.<=RSZ[bcrsjkA?>
7?4
*?'
inputs???????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????!
? ?
'__inference_model_1_layer_call_fn_43664?-.<=RSZ[bcrsjk@?=
6?3
)?&
input???????????
p 

 
? "=?:
?
0?????????
?
1?????????!?
'__inference_model_1_layer_call_fn_43939?-.<=RSZ[bcrsjk@?=
6?3
)?&
input???????????
p

 
? "=?:
?
0?????????
?
1?????????!?
'__inference_model_1_layer_call_fn_44123?-.<=RSZ[bcrsjkA?>
7?4
*?'
inputs???????????
p 

 
? "=?:
?
0?????????
?
1?????????!?
'__inference_model_1_layer_call_fn_44162?-.<=RSZ[bcrsjkA?>
7?4
*?'
inputs???????????
p

 
? "=?:
?
0?????????
?
1?????????!?
#__inference_signature_wrapper_44084?-.<=RSZ[bcrsjkA?>
? 
7?4
2
input)?&
input???????????"???
F
common.apparel_class.?+
common.apparel_class?????????
6
common.color&?#
common.color?????????!