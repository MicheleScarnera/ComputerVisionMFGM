Õ
Þ®
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8²Ú


 Adam/common.apparel_class/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/common.apparel_class/bias/v

4Adam/common.apparel_class/bias/v/Read/ReadVariableOpReadVariableOp Adam/common.apparel_class/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/common.apparel_class/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/common.apparel_class/kernel/v

6Adam/common.apparel_class/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/common.apparel_class/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_main_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/dense_main_last/bias/v

/Adam/dense_main_last/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_main_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/dense_main_last/kernel/v

1Adam/dense_main_last/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_main_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_main_2/bias/v

,Adam/dense_main_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_main_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/dense_main_2/kernel/v

.Adam/dense_main_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_main_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_main_1/bias/v

,Adam/dense_main_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_main_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*+
shared_nameAdam/dense_main_1/kernel/v

.Adam/dense_main_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/kernel/v* 
_output_shapes
:
@*
dtype0

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

Adam/conv_main_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_main_3/kernel/v

-Adam/conv_main_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/kernel/v*&
_output_shapes
: *
dtype0

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

Adam/conv_main_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_2/kernel/v

-Adam/conv_main_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/kernel/v*&
_output_shapes
:*
dtype0

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

Adam/conv_main_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_1/kernel/v

-Adam/conv_main_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_main_1/kernel/v*&
_output_shapes
:*
dtype0

 Adam/common.apparel_class/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/common.apparel_class/bias/m

4Adam/common.apparel_class/bias/m/Read/ReadVariableOpReadVariableOp Adam/common.apparel_class/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/common.apparel_class/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/common.apparel_class/kernel/m

6Adam/common.apparel_class/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/common.apparel_class/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_main_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/dense_main_last/bias/m

/Adam/dense_main_last/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_main_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/dense_main_last/kernel/m

1Adam/dense_main_last/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_last/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_main_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_main_2/bias/m

,Adam/dense_main_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_main_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/dense_main_2/kernel/m

.Adam/dense_main_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_main_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_main_1/bias/m

,Adam/dense_main_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_main_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*+
shared_nameAdam/dense_main_1/kernel/m

.Adam/dense_main_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_main_1/kernel/m* 
_output_shapes
:
@*
dtype0

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

Adam/conv_main_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_main_3/kernel/m

-Adam/conv_main_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_3/kernel/m*&
_output_shapes
: *
dtype0

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

Adam/conv_main_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_2/kernel/m

-Adam/conv_main_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_main_2/kernel/m*&
_output_shapes
:*
dtype0

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

Adam/conv_main_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_main_1/kernel/m

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

common.apparel_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecommon.apparel_class/bias

-common.apparel_class/bias/Read/ReadVariableOpReadVariableOpcommon.apparel_class/bias*
_output_shapes
:*
dtype0

common.apparel_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namecommon.apparel_class/kernel

/common.apparel_class/kernel/Read/ReadVariableOpReadVariableOpcommon.apparel_class/kernel*
_output_shapes
:	*
dtype0

dense_main_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedense_main_last/bias
z
(dense_main_last/bias/Read/ReadVariableOpReadVariableOpdense_main_last/bias*
_output_shapes	
:*
dtype0

dense_main_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namedense_main_last/kernel

*dense_main_last/kernel/Read/ReadVariableOpReadVariableOpdense_main_last/kernel* 
_output_shapes
:
*
dtype0
{
dense_main_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namedense_main_2/bias
t
%dense_main_2/bias/Read/ReadVariableOpReadVariableOpdense_main_2/bias*
_output_shapes	
:*
dtype0

dense_main_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namedense_main_2/kernel
}
'dense_main_2/kernel/Read/ReadVariableOpReadVariableOpdense_main_2/kernel* 
_output_shapes
:
*
dtype0
{
dense_main_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namedense_main_1/bias
t
%dense_main_1/bias/Read/ReadVariableOpReadVariableOpdense_main_1/bias*
_output_shapes	
:*
dtype0

dense_main_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*$
shared_namedense_main_1/kernel
}
'dense_main_1/kernel/Read/ReadVariableOpReadVariableOpdense_main_1/kernel* 
_output_shapes
:
@*
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

conv_main_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv_main_3/kernel

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

conv_main_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_main_2/kernel

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

conv_main_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_main_1/kernel

&conv_main_1/kernel/Read/ReadVariableOpReadVariableOpconv_main_1/kernel*&
_output_shapes
:*
dtype0

serving_default_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
ö
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv_main_1/kernelconv_main_1/biasconv_main_2/kernelconv_main_2/biasconv_main_3/kernelconv_main_3/biasdense_main_1/kerneldense_main_1/biasdense_main_2/kerneldense_main_2/biasdense_main_last/kerneldense_main_last/biascommon.apparel_class/kernelcommon.apparel_class/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_21468

NoOpNoOp
Ín
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*n
valueþmBûm Bôm
ª
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
È
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
È
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op*

/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
È
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*

>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
¥
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J_random_generator* 
¦
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
¦
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
¦
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias*
¦
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
j
0
1
,2
-3
;4
<5
Q6
R7
Y8
Z9
a10
b11
i12
j13*
j
0
1
,2
-3
;4
<5
Q6
R7
Y8
Z9
a10
b11
i12
j13*
* 
°
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 
* 
Ü
xiter

ybeta_1

zbeta_2
	{decay
|learning_ratemØmÙ,mÚ-mÛ;mÜ<mÝQmÞRmßYmàZmáamâbmãimäjmåvævç,vè-vé;vê<vëQvìRvíYvîZvïavðbvñivòjvó*
* 

}serving_default* 

0
1*

0
1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEconv_main_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

,0
-1*

,0
-1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEconv_main_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

;0
<1*

;0
<1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

trace_0* 

 trace_0* 
b\
VARIABLE_VALUEconv_main_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_main_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

¦trace_0* 

§trace_0* 
* 
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

­trace_0
®trace_1* 

¯trace_0
°trace_1* 
* 

Q0
R1*

Q0
R1*
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

¶trace_0* 

·trace_0* 
c]
VARIABLE_VALUEdense_main_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdense_main_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

½trace_0* 

¾trace_0* 
c]
VARIABLE_VALUEdense_main_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdense_main_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

a0
b1*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

Ätrace_0* 

Åtrace_0* 
f`
VARIABLE_VALUEdense_main_last/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEdense_main_last/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

i0
j1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
ke
VARIABLE_VALUEcommon.apparel_class/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcommon.apparel_class/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
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
11*

Í0
Î1*
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
<
Ï	variables
Ð	keras_api

Ñtotal

Òcount*
M
Ó	variables
Ô	keras_api

Õtotal

Öcount
×
_fn_kwargs*

Ñ0
Ò1*

Ï	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Õ0
Ö1*

Ó	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUEAdam/conv_main_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv_main_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv_main_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_main_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_main_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_last/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_last/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/common.apparel_class/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/common.apparel_class/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv_main_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv_main_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv_main_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv_main_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_main_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_main_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_last/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_main_last/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/common.apparel_class/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/common.apparel_class/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
í
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&conv_main_1/kernel/Read/ReadVariableOp$conv_main_1/bias/Read/ReadVariableOp&conv_main_2/kernel/Read/ReadVariableOp$conv_main_2/bias/Read/ReadVariableOp&conv_main_3/kernel/Read/ReadVariableOp$conv_main_3/bias/Read/ReadVariableOp'dense_main_1/kernel/Read/ReadVariableOp%dense_main_1/bias/Read/ReadVariableOp'dense_main_2/kernel/Read/ReadVariableOp%dense_main_2/bias/Read/ReadVariableOp*dense_main_last/kernel/Read/ReadVariableOp(dense_main_last/bias/Read/ReadVariableOp/common.apparel_class/kernel/Read/ReadVariableOp-common.apparel_class/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/conv_main_1/kernel/m/Read/ReadVariableOp+Adam/conv_main_1/bias/m/Read/ReadVariableOp-Adam/conv_main_2/kernel/m/Read/ReadVariableOp+Adam/conv_main_2/bias/m/Read/ReadVariableOp-Adam/conv_main_3/kernel/m/Read/ReadVariableOp+Adam/conv_main_3/bias/m/Read/ReadVariableOp.Adam/dense_main_1/kernel/m/Read/ReadVariableOp,Adam/dense_main_1/bias/m/Read/ReadVariableOp.Adam/dense_main_2/kernel/m/Read/ReadVariableOp,Adam/dense_main_2/bias/m/Read/ReadVariableOp1Adam/dense_main_last/kernel/m/Read/ReadVariableOp/Adam/dense_main_last/bias/m/Read/ReadVariableOp6Adam/common.apparel_class/kernel/m/Read/ReadVariableOp4Adam/common.apparel_class/bias/m/Read/ReadVariableOp-Adam/conv_main_1/kernel/v/Read/ReadVariableOp+Adam/conv_main_1/bias/v/Read/ReadVariableOp-Adam/conv_main_2/kernel/v/Read/ReadVariableOp+Adam/conv_main_2/bias/v/Read/ReadVariableOp-Adam/conv_main_3/kernel/v/Read/ReadVariableOp+Adam/conv_main_3/bias/v/Read/ReadVariableOp.Adam/dense_main_1/kernel/v/Read/ReadVariableOp,Adam/dense_main_1/bias/v/Read/ReadVariableOp.Adam/dense_main_2/kernel/v/Read/ReadVariableOp,Adam/dense_main_2/bias/v/Read/ReadVariableOp1Adam/dense_main_last/kernel/v/Read/ReadVariableOp/Adam/dense_main_last/bias/v/Read/ReadVariableOp6Adam/common.apparel_class/kernel/v/Read/ReadVariableOp4Adam/common.apparel_class/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_22031
ì
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_main_1/kernelconv_main_1/biasconv_main_2/kernelconv_main_2/biasconv_main_3/kernelconv_main_3/biasdense_main_1/kerneldense_main_1/biasdense_main_2/kerneldense_main_2/biasdense_main_last/kerneldense_main_last/biascommon.apparel_class/kernelcommon.apparel_class/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv_main_1/kernel/mAdam/conv_main_1/bias/mAdam/conv_main_2/kernel/mAdam/conv_main_2/bias/mAdam/conv_main_3/kernel/mAdam/conv_main_3/bias/mAdam/dense_main_1/kernel/mAdam/dense_main_1/bias/mAdam/dense_main_2/kernel/mAdam/dense_main_2/bias/mAdam/dense_main_last/kernel/mAdam/dense_main_last/bias/m"Adam/common.apparel_class/kernel/m Adam/common.apparel_class/bias/mAdam/conv_main_1/kernel/vAdam/conv_main_1/bias/vAdam/conv_main_2/kernel/vAdam/conv_main_2/bias/vAdam/conv_main_3/kernel/vAdam/conv_main_3/bias/vAdam/dense_main_1/kernel/vAdam/dense_main_1/bias/vAdam/dense_main_2/kernel/vAdam/dense_main_2/bias/vAdam/dense_main_last/kernel/vAdam/dense_main_last/bias/v"Adam/common.apparel_class/kernel/v Adam/common.apparel_class/bias/v*?
Tin8
624*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_22194ÁÛ
ñ
 
+__inference_conv_main_2_layer_call_fn_21696

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ü
e
,__inference_dropout_main_layer_call_fn_21758

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_21160p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

e
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
J
.__inference_maxpool_main_1_layer_call_fn_21682

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

,__inference_dense_main_2_layer_call_fn_21804

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
ú
%__inference_model_layer_call_fn_21341	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
î
ú
%__inference_model_layer_call_fn_21100	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¯


O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21855

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
J
.__inference_maxpool_main_2_layer_call_fn_21712

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K

@__inference_model_layer_call_and_return_conditional_losses_21592

inputsD
*conv_main_1_conv2d_readvariableop_resource:9
+conv_main_1_biasadd_readvariableop_resource:D
*conv_main_2_conv2d_readvariableop_resource:9
+conv_main_2_biasadd_readvariableop_resource:D
*conv_main_3_conv2d_readvariableop_resource: 9
+conv_main_3_biasadd_readvariableop_resource: ?
+dense_main_1_matmul_readvariableop_resource:
@;
,dense_main_1_biasadd_readvariableop_resource:	?
+dense_main_2_matmul_readvariableop_resource:
;
,dense_main_2_biasadd_readvariableop_resource:	B
.dense_main_last_matmul_readvariableop_resource:
>
/dense_main_last_biasadd_readvariableop_resource:	F
3common_apparel_class_matmul_readvariableop_resource:	B
4common_apparel_class_biasadd_readvariableop_resource:
identity¢+common.apparel_class/BiasAdd/ReadVariableOp¢*common.apparel_class/MatMul/ReadVariableOp¢"conv_main_1/BiasAdd/ReadVariableOp¢!conv_main_1/Conv2D/ReadVariableOp¢"conv_main_2/BiasAdd/ReadVariableOp¢!conv_main_2/Conv2D/ReadVariableOp¢"conv_main_3/BiasAdd/ReadVariableOp¢!conv_main_3/Conv2D/ReadVariableOp¢#dense_main_1/BiasAdd/ReadVariableOp¢"dense_main_1/MatMul/ReadVariableOp¢#dense_main_2/BiasAdd/ReadVariableOp¢"dense_main_2/MatMul/ReadVariableOp¢&dense_main_last/BiasAdd/ReadVariableOp¢%dense_main_last/MatMul/ReadVariableOp
!conv_main_1/Conv2D/ReadVariableOpReadVariableOp*conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0³
conv_main_1/Conv2DConv2Dinputs)conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

"conv_main_1/BiasAdd/ReadVariableOpReadVariableOp+conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
conv_main_1/BiasAddBiasAddconv_main_1/Conv2D:output:0*conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
conv_main_1/ReluReluconv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
maxpool_main_1/MaxPoolMaxPoolconv_main_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

!conv_main_2/Conv2D/ReadVariableOpReadVariableOp*conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ê
conv_main_2/Conv2DConv2Dmaxpool_main_1/MaxPool:output:0)conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

"conv_main_2/BiasAdd/ReadVariableOpReadVariableOp+conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¡
conv_main_2/BiasAddBiasAddconv_main_2/Conv2D:output:0*conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@p
conv_main_2/ReluReluconv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@®
maxpool_main_2/MaxPoolMaxPoolconv_main_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

!conv_main_3/Conv2D/ReadVariableOpReadVariableOp*conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ê
conv_main_3/Conv2DConv2Dmaxpool_main_2/MaxPool:output:0)conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

"conv_main_3/BiasAdd/ReadVariableOpReadVariableOp+conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¡
conv_main_3/BiasAddBiasAddconv_main_3/Conv2D:output:0*conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
conv_main_3/ReluReluconv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten/ReshapeReshapeconv_main_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
dropout_main/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"dense_main_1/MatMul/ReadVariableOpReadVariableOp+dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
dense_main_1/MatMulMatMuldropout_main/Identity:output:0*dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_main_1/BiasAdd/ReadVariableOpReadVariableOp,dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_main_1/BiasAddBiasAdddense_main_1/MatMul:product:0+dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
dense_main_1/ReluReludense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_main_2/MatMul/ReadVariableOpReadVariableOp+dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_main_2/MatMulMatMuldense_main_1/Relu:activations:0*dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_main_2/BiasAdd/ReadVariableOpReadVariableOp,dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_main_2/BiasAddBiasAdddense_main_2/MatMul:product:0+dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
dense_main_2/ReluReludense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_main_last/MatMul/ReadVariableOpReadVariableOp.dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0£
dense_main_last/MatMulMatMuldense_main_2/Relu:activations:0-dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&dense_main_last/BiasAdd/ReadVariableOpReadVariableOp/dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
dense_main_last/BiasAddBiasAdd dense_main_last/MatMul:product:0.dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dense_main_last/TanhTanh dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*common.apparel_class/MatMul/ReadVariableOpReadVariableOp3common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¥
common.apparel_class/MatMulMatMuldense_main_last/Tanh:y:02common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp4common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
common.apparel_class/BiasAddBiasAdd%common.apparel_class/MatMul:product:03common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
common.apparel_class/SoftmaxSoftmax%common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp,^common.apparel_class/BiasAdd/ReadVariableOp+^common.apparel_class/MatMul/ReadVariableOp#^conv_main_1/BiasAdd/ReadVariableOp"^conv_main_1/Conv2D/ReadVariableOp#^conv_main_2/BiasAdd/ReadVariableOp"^conv_main_2/Conv2D/ReadVariableOp#^conv_main_3/BiasAdd/ReadVariableOp"^conv_main_3/Conv2D/ReadVariableOp$^dense_main_1/BiasAdd/ReadVariableOp#^dense_main_1/MatMul/ReadVariableOp$^dense_main_2/BiasAdd/ReadVariableOp#^dense_main_2/MatMul/ReadVariableOp'^dense_main_last/BiasAdd/ReadVariableOp&^dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2Z
+common.apparel_class/BiasAdd/ReadVariableOp+common.apparel_class/BiasAdd/ReadVariableOp2X
*common.apparel_class/MatMul/ReadVariableOp*common.apparel_class/MatMul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
^
B__inference_flatten_layer_call_and_return_conditional_losses_20991

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

e
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_21717

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
¢
@__inference_model_layer_call_and_return_conditional_losses_21069

inputs+
conv_main_1_20944:
conv_main_1_20946:+
conv_main_2_20962:
conv_main_2_20964:+
conv_main_3_20980: 
conv_main_3_20982: &
dense_main_1_21012:
@!
dense_main_1_21014:	&
dense_main_2_21029:
!
dense_main_2_21031:	)
dense_main_last_21046:
$
dense_main_last_21048:	-
common_apparel_class_21063:	(
common_apparel_class_21065:
identity¢,common.apparel_class/StatefulPartitionedCall¢#conv_main_1/StatefulPartitionedCall¢#conv_main_2/StatefulPartitionedCall¢#conv_main_3/StatefulPartitionedCall¢$dense_main_1/StatefulPartitionedCall¢$dense_main_2/StatefulPartitionedCall¢'dense_main_last/StatefulPartitionedCall
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_main_1_20944conv_main_1_20946*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943ô
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910¥
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_20962conv_main_2_20964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961ô
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922¥
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_20980conv_main_3_20982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979ß
flatten/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20991Ý
dropout_main/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_20998 
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall%dropout_main/PartitionedCall:output:0dense_main_1_21012dense_main_1_21014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011¨
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_21029dense_main_2_21031*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028´
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_21046dense_main_last_21048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045Ê
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_21063common_apparel_class_21065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ø
#__inference_signature_wrapper_21468	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_20901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

ÿ
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
¡
@__inference_model_layer_call_and_return_conditional_losses_21384	
input+
conv_main_1_21344:
conv_main_1_21346:+
conv_main_2_21350:
conv_main_2_21352:+
conv_main_3_21356: 
conv_main_3_21358: &
dense_main_1_21363:
@!
dense_main_1_21365:	&
dense_main_2_21368:
!
dense_main_2_21370:	)
dense_main_last_21373:
$
dense_main_last_21375:	-
common_apparel_class_21378:	(
common_apparel_class_21380:
identity¢,common.apparel_class/StatefulPartitionedCall¢#conv_main_1/StatefulPartitionedCall¢#conv_main_2/StatefulPartitionedCall¢#conv_main_3/StatefulPartitionedCall¢$dense_main_1/StatefulPartitionedCall¢$dense_main_2/StatefulPartitionedCall¢'dense_main_last/StatefulPartitionedCall
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputconv_main_1_21344conv_main_1_21346*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943ô
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910¥
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_21350conv_main_2_21352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961ô
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922¥
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_21356conv_main_3_21358*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979ß
flatten/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20991Ý
dropout_main/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_20998 
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall%dropout_main/PartitionedCall:output:0dense_main_1_21363dense_main_1_21365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011¨
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_21368dense_main_2_21370*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028´
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_21373dense_main_last_21375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045Ê
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_21378common_apparel_class_21380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Þ
e
G__inference_dropout_main_layer_call_and_return_conditional_losses_20998

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾5
È
@__inference_model_layer_call_and_return_conditional_losses_21427	
input+
conv_main_1_21387:
conv_main_1_21389:+
conv_main_2_21393:
conv_main_2_21395:+
conv_main_3_21399: 
conv_main_3_21401: &
dense_main_1_21406:
@!
dense_main_1_21408:	&
dense_main_2_21411:
!
dense_main_2_21413:	)
dense_main_last_21416:
$
dense_main_last_21418:	-
common_apparel_class_21421:	(
common_apparel_class_21423:
identity¢,common.apparel_class/StatefulPartitionedCall¢#conv_main_1/StatefulPartitionedCall¢#conv_main_2/StatefulPartitionedCall¢#conv_main_3/StatefulPartitionedCall¢$dense_main_1/StatefulPartitionedCall¢$dense_main_2/StatefulPartitionedCall¢'dense_main_last/StatefulPartitionedCall¢$dropout_main/StatefulPartitionedCall
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputconv_main_1_21387conv_main_1_21389*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943ô
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910¥
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_21393conv_main_2_21395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961ô
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922¥
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_21399conv_main_3_21401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979ß
flatten/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20991í
$dropout_main/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_21160¨
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall-dropout_main/StatefulPartitionedCall:output:0dense_main_1_21406dense_main_1_21408*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011¨
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_21411dense_main_2_21413*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028´
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_21416dense_main_last_21418*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045Ê
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_21421common_apparel_class_21423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall%^dropout_main/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall2L
$dropout_main/StatefulPartitionedCall$dropout_main/StatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ý	
f
G__inference_dropout_main_layer_call_and_return_conditional_losses_21775

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£

þ
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21835

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ùR

@__inference_model_layer_call_and_return_conditional_losses_21657

inputsD
*conv_main_1_conv2d_readvariableop_resource:9
+conv_main_1_biasadd_readvariableop_resource:D
*conv_main_2_conv2d_readvariableop_resource:9
+conv_main_2_biasadd_readvariableop_resource:D
*conv_main_3_conv2d_readvariableop_resource: 9
+conv_main_3_biasadd_readvariableop_resource: ?
+dense_main_1_matmul_readvariableop_resource:
@;
,dense_main_1_biasadd_readvariableop_resource:	?
+dense_main_2_matmul_readvariableop_resource:
;
,dense_main_2_biasadd_readvariableop_resource:	B
.dense_main_last_matmul_readvariableop_resource:
>
/dense_main_last_biasadd_readvariableop_resource:	F
3common_apparel_class_matmul_readvariableop_resource:	B
4common_apparel_class_biasadd_readvariableop_resource:
identity¢+common.apparel_class/BiasAdd/ReadVariableOp¢*common.apparel_class/MatMul/ReadVariableOp¢"conv_main_1/BiasAdd/ReadVariableOp¢!conv_main_1/Conv2D/ReadVariableOp¢"conv_main_2/BiasAdd/ReadVariableOp¢!conv_main_2/Conv2D/ReadVariableOp¢"conv_main_3/BiasAdd/ReadVariableOp¢!conv_main_3/Conv2D/ReadVariableOp¢#dense_main_1/BiasAdd/ReadVariableOp¢"dense_main_1/MatMul/ReadVariableOp¢#dense_main_2/BiasAdd/ReadVariableOp¢"dense_main_2/MatMul/ReadVariableOp¢&dense_main_last/BiasAdd/ReadVariableOp¢%dense_main_last/MatMul/ReadVariableOp
!conv_main_1/Conv2D/ReadVariableOpReadVariableOp*conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0³
conv_main_1/Conv2DConv2Dinputs)conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

"conv_main_1/BiasAdd/ReadVariableOpReadVariableOp+conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
conv_main_1/BiasAddBiasAddconv_main_1/Conv2D:output:0*conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
conv_main_1/ReluReluconv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
maxpool_main_1/MaxPoolMaxPoolconv_main_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

!conv_main_2/Conv2D/ReadVariableOpReadVariableOp*conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ê
conv_main_2/Conv2DConv2Dmaxpool_main_1/MaxPool:output:0)conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

"conv_main_2/BiasAdd/ReadVariableOpReadVariableOp+conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¡
conv_main_2/BiasAddBiasAddconv_main_2/Conv2D:output:0*conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@p
conv_main_2/ReluReluconv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@®
maxpool_main_2/MaxPoolMaxPoolconv_main_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

!conv_main_3/Conv2D/ReadVariableOpReadVariableOp*conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ê
conv_main_3/Conv2DConv2Dmaxpool_main_2/MaxPool:output:0)conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

"conv_main_3/BiasAdd/ReadVariableOpReadVariableOp+conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¡
conv_main_3/BiasAddBiasAddconv_main_3/Conv2D:output:0*conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
conv_main_3/ReluReluconv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten/ReshapeReshapeconv_main_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dropout_main/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_main/dropout/MulMulflatten/Reshape:output:0#dropout_main/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
dropout_main/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:§
1dropout_main/dropout/random_uniform/RandomUniformRandomUniform#dropout_main/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0h
#dropout_main/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Î
!dropout_main/dropout/GreaterEqualGreaterEqual:dropout_main/dropout/random_uniform/RandomUniform:output:0,dropout_main/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_main/dropout/CastCast%dropout_main/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_main/dropout/Mul_1Muldropout_main/dropout/Mul:z:0dropout_main/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"dense_main_1/MatMul/ReadVariableOpReadVariableOp+dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
dense_main_1/MatMulMatMuldropout_main/dropout/Mul_1:z:0*dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_main_1/BiasAdd/ReadVariableOpReadVariableOp,dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_main_1/BiasAddBiasAdddense_main_1/MatMul:product:0+dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
dense_main_1/ReluReludense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_main_2/MatMul/ReadVariableOpReadVariableOp+dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_main_2/MatMulMatMuldense_main_1/Relu:activations:0*dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_main_2/BiasAdd/ReadVariableOpReadVariableOp,dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_main_2/BiasAddBiasAdddense_main_2/MatMul:product:0+dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
dense_main_2/ReluReludense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_main_last/MatMul/ReadVariableOpReadVariableOp.dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0£
dense_main_last/MatMulMatMuldense_main_2/Relu:activations:0-dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&dense_main_last/BiasAdd/ReadVariableOpReadVariableOp/dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
dense_main_last/BiasAddBiasAdd dense_main_last/MatMul:product:0.dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dense_main_last/TanhTanh dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*common.apparel_class/MatMul/ReadVariableOpReadVariableOp3common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¥
common.apparel_class/MatMulMatMuldense_main_last/Tanh:y:02common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp4common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
common.apparel_class/BiasAddBiasAdd%common.apparel_class/MatMul:product:03common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
common.apparel_class/SoftmaxSoftmax%common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp,^common.apparel_class/BiasAdd/ReadVariableOp+^common.apparel_class/MatMul/ReadVariableOp#^conv_main_1/BiasAdd/ReadVariableOp"^conv_main_1/Conv2D/ReadVariableOp#^conv_main_2/BiasAdd/ReadVariableOp"^conv_main_2/Conv2D/ReadVariableOp#^conv_main_3/BiasAdd/ReadVariableOp"^conv_main_3/Conv2D/ReadVariableOp$^dense_main_1/BiasAdd/ReadVariableOp#^dense_main_1/MatMul/ReadVariableOp$^dense_main_2/BiasAdd/ReadVariableOp#^dense_main_2/MatMul/ReadVariableOp'^dense_main_last/BiasAdd/ReadVariableOp&^dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2Z
+common.apparel_class/BiasAdd/ReadVariableOp+common.apparel_class/BiasAdd/ReadVariableOp2X
*common.apparel_class/MatMul/ReadVariableOp*common.apparel_class/MatMul/ReadVariableOp2H
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
 
+__inference_conv_main_1_layer_call_fn_21666

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
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
:ÿÿÿÿÿÿÿÿÿ@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ñ
û
%__inference_model_layer_call_fn_21534

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
û
%__inference_model_layer_call_fn_21501

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
@
	unknown_6:	
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

,__inference_dense_main_1_layer_call_fn_21784

inputs
unknown:
@
	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ
 
+__inference_conv_main_3_layer_call_fn_21726

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
H
,__inference_dropout_main_layer_call_fn_21753

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_20998a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
F__inference_conv_main_3_layer_call_and_return_conditional_losses_21737

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011

inputs2
matmul_readvariableop_resource:
@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª

û
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21795

inputs2
matmul_readvariableop_resource:
@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv_main_2_layer_call_and_return_conditional_losses_21707

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
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
:ÿÿÿÿÿÿÿÿÿ@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ä
^
B__inference_flatten_layer_call_and_return_conditional_losses_21748

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

e
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_21687

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õi
¹
__inference__traced_save_22031
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
4savev2_common_apparel_class_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
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
;savev2_adam_common_apparel_class_bias_m_read_readvariableop8
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
;savev2_adam_common_apparel_class_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueüBù4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B å
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_conv_main_1_kernel_read_readvariableop+savev2_conv_main_1_bias_read_readvariableop-savev2_conv_main_2_kernel_read_readvariableop+savev2_conv_main_2_bias_read_readvariableop-savev2_conv_main_3_kernel_read_readvariableop+savev2_conv_main_3_bias_read_readvariableop.savev2_dense_main_1_kernel_read_readvariableop,savev2_dense_main_1_bias_read_readvariableop.savev2_dense_main_2_kernel_read_readvariableop,savev2_dense_main_2_bias_read_readvariableop1savev2_dense_main_last_kernel_read_readvariableop/savev2_dense_main_last_bias_read_readvariableop6savev2_common_apparel_class_kernel_read_readvariableop4savev2_common_apparel_class_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_conv_main_1_kernel_m_read_readvariableop2savev2_adam_conv_main_1_bias_m_read_readvariableop4savev2_adam_conv_main_2_kernel_m_read_readvariableop2savev2_adam_conv_main_2_bias_m_read_readvariableop4savev2_adam_conv_main_3_kernel_m_read_readvariableop2savev2_adam_conv_main_3_bias_m_read_readvariableop5savev2_adam_dense_main_1_kernel_m_read_readvariableop3savev2_adam_dense_main_1_bias_m_read_readvariableop5savev2_adam_dense_main_2_kernel_m_read_readvariableop3savev2_adam_dense_main_2_bias_m_read_readvariableop8savev2_adam_dense_main_last_kernel_m_read_readvariableop6savev2_adam_dense_main_last_bias_m_read_readvariableop=savev2_adam_common_apparel_class_kernel_m_read_readvariableop;savev2_adam_common_apparel_class_bias_m_read_readvariableop4savev2_adam_conv_main_1_kernel_v_read_readvariableop2savev2_adam_conv_main_1_bias_v_read_readvariableop4savev2_adam_conv_main_2_kernel_v_read_readvariableop2savev2_adam_conv_main_2_bias_v_read_readvariableop4savev2_adam_conv_main_3_kernel_v_read_readvariableop2savev2_adam_conv_main_3_bias_v_read_readvariableop5savev2_adam_dense_main_1_kernel_v_read_readvariableop3savev2_adam_dense_main_1_bias_v_read_readvariableop5savev2_adam_dense_main_2_kernel_v_read_readvariableop3savev2_adam_dense_main_2_bias_v_read_readvariableop8savev2_adam_dense_main_last_kernel_v_read_readvariableop6savev2_adam_dense_main_last_bias_v_read_readvariableop=savev2_adam_common_apparel_class_kernel_v_read_readvariableop;savev2_adam_common_apparel_class_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*á
_input_shapesÏ
Ì: ::::: : :
@::
::
::	:: : : : : : : : : ::::: : :
@::
::
::	:::::: : :
@::
::
::	:: 2(
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
@:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
@:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::%$!

_output_shapes
:	: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
: : +

_output_shapes
: :&,"
 
_output_shapes
:
@:!-

_output_shapes	
::&."
 
_output_shapes
:
:!/

_output_shapes	
::&0"
 
_output_shapes
:
:!1

_output_shapes	
::%2!

_output_shapes
:	: 3

_output_shapes
::4

_output_shapes
: 
Á5
É
@__inference_model_layer_call_and_return_conditional_losses_21277

inputs+
conv_main_1_21237:
conv_main_1_21239:+
conv_main_2_21243:
conv_main_2_21245:+
conv_main_3_21249: 
conv_main_3_21251: &
dense_main_1_21256:
@!
dense_main_1_21258:	&
dense_main_2_21261:
!
dense_main_2_21263:	)
dense_main_last_21266:
$
dense_main_last_21268:	-
common_apparel_class_21271:	(
common_apparel_class_21273:
identity¢,common.apparel_class/StatefulPartitionedCall¢#conv_main_1/StatefulPartitionedCall¢#conv_main_2/StatefulPartitionedCall¢#conv_main_3/StatefulPartitionedCall¢$dense_main_1/StatefulPartitionedCall¢$dense_main_2/StatefulPartitionedCall¢'dense_main_last/StatefulPartitionedCall¢$dropout_main/StatefulPartitionedCall
#conv_main_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_main_1_21237conv_main_1_21239*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_1_layer_call_and_return_conditional_losses_20943ô
maxpool_main_1/PartitionedCallPartitionedCall,conv_main_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_20910¥
#conv_main_2/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_1/PartitionedCall:output:0conv_main_2_21243conv_main_2_21245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_2_layer_call_and_return_conditional_losses_20961ô
maxpool_main_2/PartitionedCallPartitionedCall,conv_main_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_20922¥
#conv_main_3/StatefulPartitionedCallStatefulPartitionedCall'maxpool_main_2/PartitionedCall:output:0conv_main_3_21249conv_main_3_21251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv_main_3_layer_call_and_return_conditional_losses_20979ß
flatten/PartitionedCallPartitionedCall,conv_main_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20991í
$dropout_main/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_main_layer_call_and_return_conditional_losses_21160¨
$dense_main_1/StatefulPartitionedCallStatefulPartitionedCall-dropout_main/StatefulPartitionedCall:output:0dense_main_1_21256dense_main_1_21258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21011¨
$dense_main_2/StatefulPartitionedCallStatefulPartitionedCall-dense_main_1/StatefulPartitionedCall:output:0dense_main_2_21261dense_main_2_21263*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028´
'dense_main_last/StatefulPartitionedCallStatefulPartitionedCall-dense_main_2/StatefulPartitionedCall:output:0dense_main_last_21266dense_main_last_21268*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045Ê
,common.apparel_class/StatefulPartitionedCallStatefulPartitionedCall0dense_main_last/StatefulPartitionedCall:output:0common_apparel_class_21271common_apparel_class_21273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062
IdentityIdentity5common.apparel_class/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^common.apparel_class/StatefulPartitionedCall$^conv_main_1/StatefulPartitionedCall$^conv_main_2/StatefulPartitionedCall$^conv_main_3/StatefulPartitionedCall%^dense_main_1/StatefulPartitionedCall%^dense_main_2/StatefulPartitionedCall(^dense_main_last/StatefulPartitionedCall%^dropout_main/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2\
,common.apparel_class/StatefulPartitionedCall,common.apparel_class/StatefulPartitionedCall2J
#conv_main_1/StatefulPartitionedCall#conv_main_1/StatefulPartitionedCall2J
#conv_main_2/StatefulPartitionedCall#conv_main_2/StatefulPartitionedCall2J
#conv_main_3/StatefulPartitionedCall#conv_main_3/StatefulPartitionedCall2L
$dense_main_1/StatefulPartitionedCall$dense_main_1/StatefulPartitionedCall2L
$dense_main_2/StatefulPartitionedCall$dense_main_2/StatefulPartitionedCall2R
'dense_main_last/StatefulPartitionedCall'dense_main_last/StatefulPartitionedCall2L
$dropout_main/StatefulPartitionedCall$dropout_main/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv_main_1_layer_call_and_return_conditional_losses_21677

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿R

 __inference__wrapped_model_20901	
inputJ
0model_conv_main_1_conv2d_readvariableop_resource:?
1model_conv_main_1_biasadd_readvariableop_resource:J
0model_conv_main_2_conv2d_readvariableop_resource:?
1model_conv_main_2_biasadd_readvariableop_resource:J
0model_conv_main_3_conv2d_readvariableop_resource: ?
1model_conv_main_3_biasadd_readvariableop_resource: E
1model_dense_main_1_matmul_readvariableop_resource:
@A
2model_dense_main_1_biasadd_readvariableop_resource:	E
1model_dense_main_2_matmul_readvariableop_resource:
A
2model_dense_main_2_biasadd_readvariableop_resource:	H
4model_dense_main_last_matmul_readvariableop_resource:
D
5model_dense_main_last_biasadd_readvariableop_resource:	L
9model_common_apparel_class_matmul_readvariableop_resource:	H
:model_common_apparel_class_biasadd_readvariableop_resource:
identity¢1model/common.apparel_class/BiasAdd/ReadVariableOp¢0model/common.apparel_class/MatMul/ReadVariableOp¢(model/conv_main_1/BiasAdd/ReadVariableOp¢'model/conv_main_1/Conv2D/ReadVariableOp¢(model/conv_main_2/BiasAdd/ReadVariableOp¢'model/conv_main_2/Conv2D/ReadVariableOp¢(model/conv_main_3/BiasAdd/ReadVariableOp¢'model/conv_main_3/Conv2D/ReadVariableOp¢)model/dense_main_1/BiasAdd/ReadVariableOp¢(model/dense_main_1/MatMul/ReadVariableOp¢)model/dense_main_2/BiasAdd/ReadVariableOp¢(model/dense_main_2/MatMul/ReadVariableOp¢,model/dense_main_last/BiasAdd/ReadVariableOp¢+model/dense_main_last/MatMul/ReadVariableOp 
'model/conv_main_1/Conv2D/ReadVariableOpReadVariableOp0model_conv_main_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¾
model/conv_main_1/Conv2DConv2Dinput/model/conv_main_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(model/conv_main_1/BiasAdd/ReadVariableOpReadVariableOp1model_conv_main_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
model/conv_main_1/BiasAddBiasAdd!model/conv_main_1/Conv2D:output:00model/conv_main_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
model/conv_main_1/ReluRelu"model/conv_main_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
model/maxpool_main_1/MaxPoolMaxPool$model/conv_main_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
 
'model/conv_main_2/Conv2D/ReadVariableOpReadVariableOp0model_conv_main_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
model/conv_main_2/Conv2DConv2D%model/maxpool_main_1/MaxPool:output:0/model/conv_main_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

(model/conv_main_2/BiasAdd/ReadVariableOpReadVariableOp1model_conv_main_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model/conv_main_2/BiasAddBiasAdd!model/conv_main_2/Conv2D:output:00model/conv_main_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@|
model/conv_main_2/ReluRelu"model/conv_main_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@º
model/maxpool_main_2/MaxPoolMaxPool$model/conv_main_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
 
'model/conv_main_3/Conv2D/ReadVariableOpReadVariableOp0model_conv_main_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ü
model/conv_main_3/Conv2DConv2D%model/maxpool_main_2/MaxPool:output:0/model/conv_main_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model/conv_main_3/BiasAdd/ReadVariableOpReadVariableOp1model_conv_main_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model/conv_main_3/BiasAddBiasAdd!model/conv_main_3/Conv2D:output:00model/conv_main_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
model/conv_main_3/ReluRelu"model/conv_main_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
model/flatten/ReshapeReshape$model/conv_main_3/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
model/dropout_main/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model/dense_main_1/MatMul/ReadVariableOpReadVariableOp1model_dense_main_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0®
model/dense_main_1/MatMulMatMul$model/dropout_main/Identity:output:00model/dense_main_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/dense_main_1/BiasAdd/ReadVariableOpReadVariableOp2model_dense_main_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/dense_main_1/BiasAddBiasAdd#model/dense_main_1/MatMul:product:01model/dense_main_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/dense_main_1/ReluRelu#model/dense_main_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/dense_main_2/MatMul/ReadVariableOpReadVariableOp1model_dense_main_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¯
model/dense_main_2/MatMulMatMul%model/dense_main_1/Relu:activations:00model/dense_main_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/dense_main_2/BiasAdd/ReadVariableOpReadVariableOp2model_dense_main_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/dense_main_2/BiasAddBiasAdd#model/dense_main_2/MatMul:product:01model/dense_main_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/dense_main_2/ReluRelu#model/dense_main_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
+model/dense_main_last/MatMul/ReadVariableOpReadVariableOp4model_dense_main_last_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0µ
model/dense_main_last/MatMulMatMul%model/dense_main_2/Relu:activations:03model/dense_main_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,model/dense_main_last/BiasAdd/ReadVariableOpReadVariableOp5model_dense_main_last_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¹
model/dense_main_last/BiasAddBiasAdd&model/dense_main_last/MatMul:product:04model/dense_main_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
model/dense_main_last/TanhTanh&model/dense_main_last/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
0model/common.apparel_class/MatMul/ReadVariableOpReadVariableOp9model_common_apparel_class_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0·
!model/common.apparel_class/MatMulMatMulmodel/dense_main_last/Tanh:y:08model/common.apparel_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1model/common.apparel_class/BiasAdd/ReadVariableOpReadVariableOp:model_common_apparel_class_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
"model/common.apparel_class/BiasAddBiasAdd+model/common.apparel_class/MatMul:product:09model/common.apparel_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/common.apparel_class/SoftmaxSoftmax+model/common.apparel_class/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
IdentityIdentity,model/common.apparel_class/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp2^model/common.apparel_class/BiasAdd/ReadVariableOp1^model/common.apparel_class/MatMul/ReadVariableOp)^model/conv_main_1/BiasAdd/ReadVariableOp(^model/conv_main_1/Conv2D/ReadVariableOp)^model/conv_main_2/BiasAdd/ReadVariableOp(^model/conv_main_2/Conv2D/ReadVariableOp)^model/conv_main_3/BiasAdd/ReadVariableOp(^model/conv_main_3/Conv2D/ReadVariableOp*^model/dense_main_1/BiasAdd/ReadVariableOp)^model/dense_main_1/MatMul/ReadVariableOp*^model/dense_main_2/BiasAdd/ReadVariableOp)^model/dense_main_2/MatMul/ReadVariableOp-^model/dense_main_last/BiasAdd/ReadVariableOp,^model/dense_main_last/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2f
1model/common.apparel_class/BiasAdd/ReadVariableOp1model/common.apparel_class/BiasAdd/ReadVariableOp2d
0model/common.apparel_class/MatMul/ReadVariableOp0model/common.apparel_class/MatMul/ReadVariableOp2T
(model/conv_main_1/BiasAdd/ReadVariableOp(model/conv_main_1/BiasAdd/ReadVariableOp2R
'model/conv_main_1/Conv2D/ReadVariableOp'model/conv_main_1/Conv2D/ReadVariableOp2T
(model/conv_main_2/BiasAdd/ReadVariableOp(model/conv_main_2/BiasAdd/ReadVariableOp2R
'model/conv_main_2/Conv2D/ReadVariableOp'model/conv_main_2/Conv2D/ReadVariableOp2T
(model/conv_main_3/BiasAdd/ReadVariableOp(model/conv_main_3/BiasAdd/ReadVariableOp2R
'model/conv_main_3/Conv2D/ReadVariableOp'model/conv_main_3/Conv2D/ReadVariableOp2V
)model/dense_main_1/BiasAdd/ReadVariableOp)model/dense_main_1/BiasAdd/ReadVariableOp2T
(model/dense_main_1/MatMul/ReadVariableOp(model/dense_main_1/MatMul/ReadVariableOp2V
)model/dense_main_2/BiasAdd/ReadVariableOp)model/dense_main_2/BiasAdd/ReadVariableOp2T
(model/dense_main_2/MatMul/ReadVariableOp(model/dense_main_2/MatMul/ReadVariableOp2\
,model/dense_main_last/BiasAdd/ReadVariableOp,model/dense_main_last/BiasAdd/ReadVariableOp2Z
+model/dense_main_last/MatMul/ReadVariableOp+model/dense_main_last/MatMul/ReadVariableOp:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ØÎ
!
!__inference__traced_restore_22194
file_prefix=
#assignvariableop_conv_main_1_kernel:1
#assignvariableop_1_conv_main_1_bias:?
%assignvariableop_2_conv_main_2_kernel:1
#assignvariableop_3_conv_main_2_bias:?
%assignvariableop_4_conv_main_3_kernel: 1
#assignvariableop_5_conv_main_3_bias: :
&assignvariableop_6_dense_main_1_kernel:
@3
$assignvariableop_7_dense_main_1_bias:	:
&assignvariableop_8_dense_main_2_kernel:
3
$assignvariableop_9_dense_main_2_bias:	>
*assignvariableop_10_dense_main_last_kernel:
7
(assignvariableop_11_dense_main_last_bias:	B
/assignvariableop_12_common_apparel_class_kernel:	;
-assignvariableop_13_common_apparel_class_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: #
assignvariableop_21_total: #
assignvariableop_22_count: G
-assignvariableop_23_adam_conv_main_1_kernel_m:9
+assignvariableop_24_adam_conv_main_1_bias_m:G
-assignvariableop_25_adam_conv_main_2_kernel_m:9
+assignvariableop_26_adam_conv_main_2_bias_m:G
-assignvariableop_27_adam_conv_main_3_kernel_m: 9
+assignvariableop_28_adam_conv_main_3_bias_m: B
.assignvariableop_29_adam_dense_main_1_kernel_m:
@;
,assignvariableop_30_adam_dense_main_1_bias_m:	B
.assignvariableop_31_adam_dense_main_2_kernel_m:
;
,assignvariableop_32_adam_dense_main_2_bias_m:	E
1assignvariableop_33_adam_dense_main_last_kernel_m:
>
/assignvariableop_34_adam_dense_main_last_bias_m:	I
6assignvariableop_35_adam_common_apparel_class_kernel_m:	B
4assignvariableop_36_adam_common_apparel_class_bias_m:G
-assignvariableop_37_adam_conv_main_1_kernel_v:9
+assignvariableop_38_adam_conv_main_1_bias_v:G
-assignvariableop_39_adam_conv_main_2_kernel_v:9
+assignvariableop_40_adam_conv_main_2_bias_v:G
-assignvariableop_41_adam_conv_main_3_kernel_v: 9
+assignvariableop_42_adam_conv_main_3_bias_v: B
.assignvariableop_43_adam_dense_main_1_kernel_v:
@;
,assignvariableop_44_adam_dense_main_1_bias_v:	B
.assignvariableop_45_adam_dense_main_2_kernel_v:
;
,assignvariableop_46_adam_dense_main_2_bias_v:	E
1assignvariableop_47_adam_dense_main_last_kernel_v:
>
/assignvariableop_48_adam_dense_main_last_bias_v:	I
6assignvariableop_49_adam_common_apparel_class_kernel_v:	B
4assignvariableop_50_adam_common_apparel_class_bias_v:
identity_52¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9à
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueüBù4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_conv_main_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv_main_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv_main_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv_main_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv_main_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv_main_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_dense_main_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_dense_main_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_dense_main_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_dense_main_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp*assignvariableop_10_dense_main_last_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp(assignvariableop_11_dense_main_last_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_common_apparel_class_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp-assignvariableop_13_common_apparel_class_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_conv_main_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_conv_main_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_conv_main_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_conv_main_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_conv_main_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_conv_main_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_dense_main_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_dense_main_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_main_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_main_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_dense_main_last_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_dense_main_last_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_common_apparel_class_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_common_apparel_class_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_conv_main_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_conv_main_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_conv_main_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_conv_main_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_conv_main_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_conv_main_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_dense_main_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_dense_main_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_dense_main_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_dense_main_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adam_dense_main_last_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_dense_main_last_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_common_apparel_class_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_common_apparel_class_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ±	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ª

û
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21028

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

/__inference_dense_main_last_layer_call_fn_21824

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý	
f
G__inference_dropout_main_layer_call_and_return_conditional_losses_21160

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
C
'__inference_flatten_layer_call_fn_21742

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20991a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£

þ
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21045

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
4__inference_common.apparel_class_layer_call_fn_21844

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
e
G__inference_dropout_main_layer_call_and_return_conditional_losses_21763

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª

û
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21815

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21062

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*½
serving_default©
A
input8
serving_default_input:0ÿÿÿÿÿÿÿÿÿH
common.apparel_class0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ë
Á
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
¥
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op"
_tf_keras_layer
¥
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
¥
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J_random_generator"
_tf_keras_layer
»
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
»
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
»
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias"
_tf_keras_layer
»
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer

0
1
,2
-3
;4
<5
Q6
R7
Y8
Z9
a10
b11
i12
j13"
trackable_list_wrapper

0
1
,2
-3
;4
<5
Q6
R7
Y8
Z9
a10
b11
i12
j13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
É
ptrace_0
qtrace_1
rtrace_2
strace_32Þ
%__inference_model_layer_call_fn_21100
%__inference_model_layer_call_fn_21501
%__inference_model_layer_call_fn_21534
%__inference_model_layer_call_fn_21341¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
µ
ttrace_0
utrace_1
vtrace_2
wtrace_32Ê
@__inference_model_layer_call_and_return_conditional_losses_21592
@__inference_model_layer_call_and_return_conditional_losses_21657
@__inference_model_layer_call_and_return_conditional_losses_21384
@__inference_model_layer_call_and_return_conditional_losses_21427¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
ÉBÆ
 __inference__wrapped_model_20901input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë
xiter

ybeta_1

zbeta_2
	{decay
|learning_ratemØmÙ,mÚ-mÛ;mÜ<mÝQmÞRmßYmàZmáamâbmãimäjmåvævç,vè-vé;vê<vëQvìRvíYvîZvïavðbvñivòjvó"
	optimizer
 "
trackable_dict_wrapper
,
}serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_conv_main_1_layer_call_fn_21666¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_conv_main_1_layer_call_and_return_conditional_losses_21677¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
,:*2conv_main_1/kernel
:2conv_main_1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ô
trace_02Õ
.__inference_maxpool_main_1_layer_call_fn_21682¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ð
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_21687¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_conv_main_2_layer_call_fn_21696¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_conv_main_2_layer_call_and_return_conditional_losses_21707¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
,:*2conv_main_2/kernel
:2conv_main_2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ô
trace_02Õ
.__inference_maxpool_main_2_layer_call_fn_21712¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ð
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_21717¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_conv_main_3_layer_call_fn_21726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

 trace_02í
F__inference_conv_main_3_layer_call_and_return_conditional_losses_21737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0
,:* 2conv_main_3/kernel
: 2conv_main_3/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
í
¦trace_02Î
'__inference_flatten_layer_call_fn_21742¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02é
B__inference_flatten_layer_call_and_return_conditional_losses_21748¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
Í
­trace_0
®trace_12
,__inference_dropout_main_layer_call_fn_21753
,__inference_dropout_main_layer_call_fn_21758³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0z®trace_1

¯trace_0
°trace_12È
G__inference_dropout_main_layer_call_and_return_conditional_losses_21763
G__inference_dropout_main_layer_call_and_return_conditional_losses_21775³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0z°trace_1
"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ò
¶trace_02Ó
,__inference_dense_main_1_layer_call_fn_21784¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¶trace_0

·trace_02î
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21795¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0
':%
@2dense_main_1/kernel
 :2dense_main_1/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
ò
½trace_02Ó
,__inference_dense_main_2_layer_call_fn_21804¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0

¾trace_02î
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0
':%
2dense_main_2/kernel
 :2dense_main_2/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
õ
Ätrace_02Ö
/__inference_dense_main_last_layer_call_fn_21824¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÄtrace_0

Åtrace_02ñ
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0
*:(
2dense_main_last/kernel
#:!2dense_main_last/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
ú
Ëtrace_02Û
4__inference_common.apparel_class_layer_call_fn_21844¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zËtrace_0

Ìtrace_02ö
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0
.:,	2common.apparel_class/kernel
':%2common.apparel_class/bias
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
õBò
%__inference_model_layer_call_fn_21100input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
%__inference_model_layer_call_fn_21501inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
%__inference_model_layer_call_fn_21534inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
%__inference_model_layer_call_fn_21341input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_21592inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_21657inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_21384input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_21427input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÈBÅ
#__inference_signature_wrapper_21468input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv_main_1_layer_call_fn_21666inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv_main_1_layer_call_and_return_conditional_losses_21677inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
âBß
.__inference_maxpool_main_1_layer_call_fn_21682inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_21687inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv_main_2_layer_call_fn_21696inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv_main_2_layer_call_and_return_conditional_losses_21707inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
âBß
.__inference_maxpool_main_2_layer_call_fn_21712inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_21717inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv_main_3_layer_call_fn_21726inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv_main_3_layer_call_and_return_conditional_losses_21737inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÛBØ
'__inference_flatten_layer_call_fn_21742inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_flatten_layer_call_and_return_conditional_losses_21748inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ñBî
,__inference_dropout_main_layer_call_fn_21753inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
,__inference_dropout_main_layer_call_fn_21758inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_main_layer_call_and_return_conditional_losses_21763inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_main_layer_call_and_return_conditional_losses_21775inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
àBÝ
,__inference_dense_main_1_layer_call_fn_21784inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21795inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
àBÝ
,__inference_dense_main_2_layer_call_fn_21804inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21815inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ãBà
/__inference_dense_main_last_layer_call_fn_21824inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21835inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
èBå
4__inference_common.apparel_class_layer_call_fn_21844inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21855inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
Ï	variables
Ð	keras_api

Ñtotal

Òcount"
_tf_keras_metric
c
Ó	variables
Ô	keras_api

Õtotal

Öcount
×
_fn_kwargs"
_tf_keras_metric
0
Ñ0
Ò1"
trackable_list_wrapper
.
Ï	variables"
_generic_user_object
:  (2total
:  (2count
0
Õ0
Ö1"
trackable_list_wrapper
.
Ó	variables"
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
@2Adam/dense_main_1/kernel/m
%:#2Adam/dense_main_1/bias/m
,:*
2Adam/dense_main_2/kernel/m
%:#2Adam/dense_main_2/bias/m
/:-
2Adam/dense_main_last/kernel/m
(:&2Adam/dense_main_last/bias/m
3:1	2"Adam/common.apparel_class/kernel/m
,:*2 Adam/common.apparel_class/bias/m
1:/2Adam/conv_main_1/kernel/v
#:!2Adam/conv_main_1/bias/v
1:/2Adam/conv_main_2/kernel/v
#:!2Adam/conv_main_2/bias/v
1:/ 2Adam/conv_main_3/kernel/v
#:! 2Adam/conv_main_3/bias/v
,:*
@2Adam/dense_main_1/kernel/v
%:#2Adam/dense_main_1/bias/v
,:*
2Adam/dense_main_2/kernel/v
%:#2Adam/dense_main_2/bias/v
/:-
2Adam/dense_main_last/kernel/v
(:&2Adam/dense_main_last/bias/v
3:1	2"Adam/common.apparel_class/kernel/v
,:*2 Adam/common.apparel_class/bias/v¼
 __inference__wrapped_model_20901,-;<QRYZabij8¢5
.¢+
)&
inputÿÿÿÿÿÿÿÿÿ
ª "KªH
F
common.apparel_class.+
common.apparel_classÿÿÿÿÿÿÿÿÿ°
O__inference_common.apparel_class_layer_call_and_return_conditional_losses_21855]ij0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_common.apparel_class_layer_call_fn_21844Pij0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿº
F__inference_conv_main_1_layer_call_and_return_conditional_losses_21677p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv_main_1_layer_call_fn_21666c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¶
F__inference_conv_main_2_layer_call_and_return_conditional_losses_21707l,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@
 
+__inference_conv_main_2_layer_call_fn_21696_,-7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@
ª " ÿÿÿÿÿÿÿÿÿ@@¶
F__inference_conv_main_3_layer_call_and_return_conditional_losses_21737l;<7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
+__inference_conv_main_3_layer_call_fn_21726_;<7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ ©
G__inference_dense_main_1_layer_call_and_return_conditional_losses_21795^QR0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_main_1_layer_call_fn_21784QQR0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_main_2_layer_call_and_return_conditional_losses_21815^YZ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_main_2_layer_call_fn_21804QYZ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
J__inference_dense_main_last_layer_call_and_return_conditional_losses_21835^ab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_dense_main_last_layer_call_fn_21824Qab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dropout_main_layer_call_and_return_conditional_losses_21763^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ@
 ©
G__inference_dropout_main_layer_call_and_return_conditional_losses_21775^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_dropout_main_layer_call_fn_21753Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@
,__inference_dropout_main_layer_call_fn_21758Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@§
B__inference_flatten_layer_call_and_return_conditional_losses_21748a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_flatten_layer_call_fn_21742T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ@ì
I__inference_maxpool_main_1_layer_call_and_return_conditional_losses_21687R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_maxpool_main_1_layer_call_fn_21682R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_maxpool_main_2_layer_call_and_return_conditional_losses_21717R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_maxpool_main_2_layer_call_fn_21712R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
@__inference_model_layer_call_and_return_conditional_losses_21384y,-;<QRYZabij@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
@__inference_model_layer_call_and_return_conditional_losses_21427y,-;<QRYZabij@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
@__inference_model_layer_call_and_return_conditional_losses_21592z,-;<QRYZabijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
@__inference_model_layer_call_and_return_conditional_losses_21657z,-;<QRYZabijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
%__inference_model_layer_call_fn_21100l,-;<QRYZabij@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_21341l,-;<QRYZabij@¢=
6¢3
)&
inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_21501m,-;<QRYZabijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_21534m,-;<QRYZabijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
#__inference_signature_wrapper_21468 ,-;<QRYZabijA¢>
¢ 
7ª4
2
input)&
inputÿÿÿÿÿÿÿÿÿ"KªH
F
common.apparel_class.+
common.apparel_classÿÿÿÿÿÿÿÿÿ