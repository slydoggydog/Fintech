(()=>{var Ec="170";var Pu=0,al=1,Lu=2;var xh=1,bc=2,Tn=3,$n=0,Be=1,ve=2,qn=0,Zi=1,cl=2,ll=3,hl=4,Du=5,gi=100,Uu=101,Nu=102,Fu=103,Ou=104,Bu=200,zu=201,Hu=202,ku=203,Qo=204,jo=205,Vu=206,Gu=207,Wu=208,Xu=209,qu=210,Yu=211,Zu=212,$u=213,Ju=214,ta=0,ea=1,na=2,Qi=3,ia=4,sa=5,ra=6,oa=7,yh=0,Ku=1,Qu=2,Yn=0,ju=1,td=2,ed=3,wc=4,nd=5,id=6,sd=7;var vh=300,ji=301,ts=302,aa=303,ca=304,io=306,Ds=1e3,yi=1001,la=1002,rn=1003,rd=1004;var Ks=1005;var hn=1006,yo=1007;var vi=1008;var Pn=1009,Mh=1010,Sh=1011,Us=1012,Tc=1013,Mi=1014,Rn=1015,Gs=1016,Ac=1017,Rc=1018,es=1020,Eh=35902,bh=1021,wh=1022,sn=1023,Th=1024,Ah=1025,$i=1026,ns=1027,Rh=1028,Cc=1029,Ch=1030,Ic=1031;var Pc=1033,Sr=33776,Er=33777,br=33778,wr=33779,ha=35840,ua=35841,da=35842,fa=35843,pa=36196,ma=37492,ga=37496,_a=37808,xa=37809,ya=37810,va=37811,Ma=37812,Sa=37813,Ea=37814,ba=37815,wa=37816,Ta=37817,Aa=37818,Ra=37819,Ca=37820,Ia=37821,Tr=36492,Pa=36494,La=36495,Ih=36283,Da=36284,Ua=36285,Na=36286;var Ar=2300,Fa=2301,vo=2302,ul=2400,dl=2401,fl=2402;var od=3200,ad=3201;var Ph=0,cd=1,Xn="",xe="srgb",cs="srgb-linear",so="linear",ne="srgb";var Pi=7680;var pl=519,ld=512,hd=513,ud=514,Lh=515,dd=516,fd=517,pd=518,md=519,ml=35044;var gl="300 es",Cn=2e3,Rr=2001,Jn=class{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[t]===void 0&&(i[t]=[]),i[t].indexOf(e)===-1&&i[t].push(e)}hasEventListener(t,e){if(this._listeners===void 0)return!1;let i=this._listeners;return i[t]!==void 0&&i[t].indexOf(e)!==-1}removeEventListener(t,e){if(this._listeners===void 0)return;let s=this._listeners[t];if(s!==void 0){let r=s.indexOf(e);r!==-1&&s.splice(r,1)}}dispatchEvent(t){if(this._listeners===void 0)return;let i=this._listeners[t.type];if(i!==void 0){t.target=this;let s=i.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,t);t.target=null}}},Ce=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],_l=1234567,Rs=Math.PI/180,Ns=180/Math.PI;function Ri(){let n=Math.random()*4294967295|0,t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(Ce[n&255]+Ce[n>>8&255]+Ce[n>>16&255]+Ce[n>>24&255]+"-"+Ce[t&255]+Ce[t>>8&255]+"-"+Ce[t>>16&15|64]+Ce[t>>24&255]+"-"+Ce[e&63|128]+Ce[e>>8&255]+"-"+Ce[e>>16&255]+Ce[e>>24&255]+Ce[i&255]+Ce[i>>8&255]+Ce[i>>16&255]+Ce[i>>24&255]).toLowerCase()}function ye(n,t,e){return Math.max(t,Math.min(e,n))}function Lc(n,t){return(n%t+t)%t}function gd(n,t,e,i,s){return i+(n-t)*(s-i)/(e-t)}function _d(n,t,e){return n!==t?(e-n)/(t-n):0}function Cs(n,t,e){return(1-e)*n+e*t}function xd(n,t,e,i){return Cs(n,t,1-Math.exp(-e*i))}function yd(n,t=1){return t-Math.abs(Lc(n,t*2)-t)}function vd(n,t,e){return n<=t?0:n>=e?1:(n=(n-t)/(e-t),n*n*(3-2*n))}function Md(n,t,e){return n<=t?0:n>=e?1:(n=(n-t)/(e-t),n*n*n*(n*(n*6-15)+10))}function Sd(n,t){return n+Math.floor(Math.random()*(t-n+1))}function Ed(n,t){return n+Math.random()*(t-n)}function bd(n){return n*(.5-Math.random())}function wd(n){n!==void 0&&(_l=n);let t=_l+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=t+Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}function Td(n){return n*Rs}function Ad(n){return n*Ns}function Rd(n){return(n&n-1)===0&&n!==0}function Cd(n){return Math.pow(2,Math.ceil(Math.log(n)/Math.LN2))}function Id(n){return Math.pow(2,Math.floor(Math.log(n)/Math.LN2))}function Pd(n,t,e,i,s){let r=Math.cos,o=Math.sin,a=r(e/2),c=o(e/2),l=r((t+i)/2),h=o((t+i)/2),u=r((t-i)/2),d=o((t-i)/2),f=r((i-t)/2),g=o((i-t)/2);switch(s){case"XYX":n.set(a*h,c*u,c*d,a*l);break;case"YZY":n.set(c*d,a*h,c*u,a*l);break;case"ZXZ":n.set(c*u,c*d,a*h,a*l);break;case"XZX":n.set(a*h,c*g,c*f,a*l);break;case"YXY":n.set(c*f,a*h,c*g,a*l);break;case"ZYZ":n.set(c*g,c*f,a*h,a*l);break;default:console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Xi(n,t){switch(t.constructor){case Float32Array:return n;case Uint32Array:return n/4294967295;case Uint16Array:return n/65535;case Uint8Array:return n/255;case Int32Array:return Math.max(n/2147483647,-1);case Int16Array:return Math.max(n/32767,-1);case Int8Array:return Math.max(n/127,-1);default:throw new Error("Invalid component type.")}}function Le(n,t){switch(t.constructor){case Float32Array:return n;case Uint32Array:return Math.round(n*4294967295);case Uint16Array:return Math.round(n*65535);case Uint8Array:return Math.round(n*255);case Int32Array:return Math.round(n*2147483647);case Int16Array:return Math.round(n*32767);case Int8Array:return Math.round(n*127);default:throw new Error("Invalid component type.")}}var ro={DEG2RAD:Rs,RAD2DEG:Ns,generateUUID:Ri,clamp:ye,euclideanModulo:Lc,mapLinear:gd,inverseLerp:_d,lerp:Cs,damp:xd,pingpong:yd,smoothstep:vd,smootherstep:Md,randInt:Sd,randFloat:Ed,randFloatSpread:bd,seededRandom:wd,degToRad:Td,radToDeg:Ad,isPowerOfTwo:Rd,ceilPowerOfTwo:Cd,floorPowerOfTwo:Id,setQuaternionFromProperEuler:Pd,normalize:Le,denormalize:Xi},j=class n{constructor(t=0,e=0){n.prototype.isVector2=!0,this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t){return this.x+=t.x,this.y+=t.y,this}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){let e=this.x,i=this.y,s=t.elements;return this.x=s[0]*e+s[3]*i+s[6],this.y=s[1]*e+s[4]*i+s[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this}clampLength(t,e){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(t,Math.min(e,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(t){let e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;let i=this.dot(t)/e;return Math.acos(ye(i,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){let e=this.x-t.x,i=this.y-t.y;return e*e+i*i}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,i){return this.x=t.x+(e.x-t.x)*i,this.y=t.y+(e.y-t.y)*i,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){let i=Math.cos(e),s=Math.sin(e),r=this.x-t.x,o=this.y-t.y;return this.x=r*i-o*s+t.x,this.y=r*s+o*i+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}},Wt=class n{constructor(t,e,i,s,r,o,a,c,l){n.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],t!==void 0&&this.set(t,e,i,s,r,o,a,c,l)}set(t,e,i,s,r,o,a,c,l){let h=this.elements;return h[0]=t,h[1]=s,h[2]=a,h[3]=e,h[4]=r,h[5]=c,h[6]=i,h[7]=o,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){let e=this.elements,i=t.elements;return e[0]=i[0],e[1]=i[1],e[2]=i[2],e[3]=i[3],e[4]=i[4],e[5]=i[5],e[6]=i[6],e[7]=i[7],e[8]=i[8],this}extractBasis(t,e,i){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(t){let e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){let i=t.elements,s=e.elements,r=this.elements,o=i[0],a=i[3],c=i[6],l=i[1],h=i[4],u=i[7],d=i[2],f=i[5],g=i[8],_=s[0],m=s[3],p=s[6],S=s[1],M=s[4],x=s[7],D=s[2],T=s[5],R=s[8];return r[0]=o*_+a*S+c*D,r[3]=o*m+a*M+c*T,r[6]=o*p+a*x+c*R,r[1]=l*_+h*S+u*D,r[4]=l*m+h*M+u*T,r[7]=l*p+h*x+u*R,r[2]=d*_+f*S+g*D,r[5]=d*m+f*M+g*T,r[8]=d*p+f*x+g*R,this}multiplyScalar(t){let e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){let t=this.elements,e=t[0],i=t[1],s=t[2],r=t[3],o=t[4],a=t[5],c=t[6],l=t[7],h=t[8];return e*o*h-e*a*l-i*r*h+i*a*c+s*r*l-s*o*c}invert(){let t=this.elements,e=t[0],i=t[1],s=t[2],r=t[3],o=t[4],a=t[5],c=t[6],l=t[7],h=t[8],u=h*o-a*l,d=a*c-h*r,f=l*r-o*c,g=e*u+i*d+s*f;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);let _=1/g;return t[0]=u*_,t[1]=(s*l-h*i)*_,t[2]=(a*i-s*o)*_,t[3]=d*_,t[4]=(h*e-s*c)*_,t[5]=(s*r-a*e)*_,t[6]=f*_,t[7]=(i*c-l*e)*_,t[8]=(o*e-i*r)*_,this}transpose(){let t,e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){let e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,i,s,r,o,a){let c=Math.cos(r),l=Math.sin(r);return this.set(i*c,i*l,-i*(c*o+l*a)+o+t,-s*l,s*c,-s*(-l*o+c*a)+a+e,0,0,1),this}scale(t,e){return this.premultiply(Mo.makeScale(t,e)),this}rotate(t){return this.premultiply(Mo.makeRotation(-t)),this}translate(t,e){return this.premultiply(Mo.makeTranslation(t,e)),this}makeTranslation(t,e){return t.isVector2?this.set(1,0,t.x,0,1,t.y,0,0,1):this.set(1,0,t,0,1,e,0,0,1),this}makeRotation(t){let e=Math.cos(t),i=Math.sin(t);return this.set(e,-i,0,i,e,0,0,0,1),this}makeScale(t,e){return this.set(t,0,0,0,e,0,0,0,1),this}equals(t){let e=this.elements,i=t.elements;for(let s=0;s<9;s++)if(e[s]!==i[s])return!1;return!0}fromArray(t,e=0){for(let i=0;i<9;i++)this.elements[i]=t[i+e];return this}toArray(t=[],e=0){let i=this.elements;return t[e]=i[0],t[e+1]=i[1],t[e+2]=i[2],t[e+3]=i[3],t[e+4]=i[4],t[e+5]=i[5],t[e+6]=i[6],t[e+7]=i[7],t[e+8]=i[8],t}clone(){return new this.constructor().fromArray(this.elements)}},Mo=new Wt;function Dh(n){for(let t=n.length-1;t>=0;--t)if(n[t]>=65535)return!0;return!1}function Cr(n){return document.createElementNS("http://www.w3.org/1999/xhtml",n)}function Ld(){let n=Cr("canvas");return n.style.display="block",n}var xl={};function Ts(n){n in xl||(xl[n]=!0,console.warn(n))}function Dd(n,t,e){return new Promise(function(i,s){function r(){switch(n.clientWaitSync(t,n.SYNC_FLUSH_COMMANDS_BIT,0)){case n.WAIT_FAILED:s();break;case n.TIMEOUT_EXPIRED:setTimeout(r,e);break;default:i()}}setTimeout(r,e)})}function Ud(n){let t=n.elements;t[2]=.5*t[2]+.5*t[3],t[6]=.5*t[6]+.5*t[7],t[10]=.5*t[10]+.5*t[11],t[14]=.5*t[14]+.5*t[15]}function Nd(n){let t=n.elements;t[11]===-1?(t[10]=-t[10]-1,t[14]=-t[14]):(t[10]=-t[10],t[14]=-t[14]+1)}var Jt={enabled:!0,workingColorSpace:cs,spaces:{},convert:function(n,t,e){return this.enabled===!1||t===e||!t||!e||(this.spaces[t].transfer===ne&&(n.r=In(n.r),n.g=In(n.g),n.b=In(n.b)),this.spaces[t].primaries!==this.spaces[e].primaries&&(n.applyMatrix3(this.spaces[t].toXYZ),n.applyMatrix3(this.spaces[e].fromXYZ)),this.spaces[e].transfer===ne&&(n.r=Ji(n.r),n.g=Ji(n.g),n.b=Ji(n.b))),n},fromWorkingColorSpace:function(n,t){return this.convert(n,this.workingColorSpace,t)},toWorkingColorSpace:function(n,t){return this.convert(n,t,this.workingColorSpace)},getPrimaries:function(n){return this.spaces[n].primaries},getTransfer:function(n){return n===Xn?so:this.spaces[n].transfer},getLuminanceCoefficients:function(n,t=this.workingColorSpace){return n.fromArray(this.spaces[t].luminanceCoefficients)},define:function(n){Object.assign(this.spaces,n)},_getMatrix:function(n,t,e){return n.copy(this.spaces[t].toXYZ).multiply(this.spaces[e].fromXYZ)},_getDrawingBufferColorSpace:function(n){return this.spaces[n].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(n=this.workingColorSpace){return this.spaces[n].workingColorSpaceConfig.unpackColorSpace}};function In(n){return n<.04045?n*.0773993808:Math.pow(n*.9478672986+.0521327014,2.4)}function Ji(n){return n<.0031308?n*12.92:1.055*Math.pow(n,.41666)-.055}var yl=[.64,.33,.3,.6,.15,.06],vl=[.2126,.7152,.0722],Ml=[.3127,.329],Sl=new Wt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),El=new Wt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);Jt.define({[cs]:{primaries:yl,whitePoint:Ml,transfer:so,toXYZ:Sl,fromXYZ:El,luminanceCoefficients:vl,workingColorSpaceConfig:{unpackColorSpace:xe},outputColorSpaceConfig:{drawingBufferColorSpace:xe}},[xe]:{primaries:yl,whitePoint:Ml,transfer:ne,toXYZ:Sl,fromXYZ:El,luminanceCoefficients:vl,outputColorSpaceConfig:{drawingBufferColorSpace:xe}}});var Li,Oa=class{static getDataURL(t){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{Li===void 0&&(Li=Cr("canvas")),Li.width=t.width,Li.height=t.height;let i=Li.getContext("2d");t instanceof ImageData?i.putImageData(t,0,0):i.drawImage(t,0,0,t.width,t.height),e=Li}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){let e=Cr("canvas");e.width=t.width,e.height=t.height;let i=e.getContext("2d");i.drawImage(t,0,0,t.width,t.height);let s=i.getImageData(0,0,t.width,t.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=In(r[o]/255)*255;return i.putImageData(s,0,0),e}else if(t.data){let e=t.data.slice(0);for(let i=0;i<e.length;i++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[i]=Math.floor(In(e[i]/255)*255):e[i]=In(e[i]);return{data:e,width:t.width,height:t.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}},Fd=0,Ir=class{constructor(t=null){this.isSource=!0,Object.defineProperty(this,"id",{value:Fd++}),this.uuid=Ri(),this.data=t,this.dataReady=!0,this.version=0}set needsUpdate(t){t===!0&&this.version++}toJSON(t){let e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(So(s[o].image)):r.push(So(s[o]))}else r=So(s);i.url=r}return e||(t.images[this.uuid]=i),i}};function So(n){return typeof HTMLImageElement<"u"&&n instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&n instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&n instanceof ImageBitmap?Oa.getDataURL(n):n.data?{data:Array.from(n.data),width:n.width,height:n.height,type:n.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}var Od=0,We=class n extends Jn{constructor(t=n.DEFAULT_IMAGE,e=n.DEFAULT_MAPPING,i=yi,s=yi,r=hn,o=vi,a=sn,c=Pn,l=n.DEFAULT_ANISOTROPY,h=Xn){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Od++}),this.uuid=Ri(),this.name="",this.source=new Ir(t),this.mipmaps=[],this.mapping=e,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=l,this.format=a,this.internalFormat=null,this.type=c,this.offset=new j(0,0),this.repeat=new j(1,1),this.center=new j(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Wt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1,this.pmremVersion=0}get image(){return this.source.data}set image(t=null){this.source.data=t}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(t){return this.name=t.name,this.source=t.source,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.channel=t.channel,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.colorSpace=t.colorSpace,this.userData=JSON.parse(JSON.stringify(t.userData)),this.needsUpdate=!0,this}toJSON(t){let e=t===void 0||typeof t=="string";if(!e&&t.textures[this.uuid]!==void 0)return t.textures[this.uuid];let i={metadata:{version:4.6,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(t).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),e||(t.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==vh)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case Ds:t.x=t.x-Math.floor(t.x);break;case yi:t.x=t.x<0?0:1;break;case la:Math.abs(Math.floor(t.x)%2)===1?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x);break}if(t.y<0||t.y>1)switch(this.wrapT){case Ds:t.y=t.y-Math.floor(t.y);break;case yi:t.y=t.y<0?0:1;break;case la:Math.abs(Math.floor(t.y)%2)===1?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y);break}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){t===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(t){t===!0&&this.pmremVersion++}};We.DEFAULT_IMAGE=null;We.DEFAULT_MAPPING=vh;We.DEFAULT_ANISOTROPY=1;var ie=class n{constructor(t=0,e=0,i=0,s=1){n.prototype.isVector4=!0,this.x=t,this.y=e,this.z=i,this.w=s}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,i,s){return this.x=t,this.y=e,this.z=i,this.w=s,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=t.w!==void 0?t.w:1,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){let e=this.x,i=this.y,s=this.z,r=this.w,o=t.elements;return this.x=o[0]*e+o[4]*i+o[8]*s+o[12]*r,this.y=o[1]*e+o[5]*i+o[9]*s+o[13]*r,this.z=o[2]*e+o[6]*i+o[10]*s+o[14]*r,this.w=o[3]*e+o[7]*i+o[11]*s+o[15]*r,this}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this.w/=t.w,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);let e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,i,s,r,c=t.elements,l=c[0],h=c[4],u=c[8],d=c[1],f=c[5],g=c[9],_=c[2],m=c[6],p=c[10];if(Math.abs(h-d)<.01&&Math.abs(u-_)<.01&&Math.abs(g-m)<.01){if(Math.abs(h+d)<.1&&Math.abs(u+_)<.1&&Math.abs(g+m)<.1&&Math.abs(l+f+p-3)<.1)return this.set(1,0,0,0),this;e=Math.PI;let M=(l+1)/2,x=(f+1)/2,D=(p+1)/2,T=(h+d)/4,R=(u+_)/4,C=(g+m)/4;return M>x&&M>D?M<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(M),s=T/i,r=R/i):x>D?x<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(x),i=T/s,r=C/s):D<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(D),i=R/r,s=C/r),this.set(i,s,r,e),this}let S=Math.sqrt((m-g)*(m-g)+(u-_)*(u-_)+(d-h)*(d-h));return Math.abs(S)<.001&&(S=1),this.x=(m-g)/S,this.y=(u-_)/S,this.z=(d-h)/S,this.w=Math.acos((l+f+p-1)/2),this}setFromMatrixPosition(t){let e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this.w=e[15],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this.w=Math.max(t.w,Math.min(e.w,this.w)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this.w=Math.max(t,Math.min(e,this.w)),this}clampLength(t,e){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(t,Math.min(e,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,i){return this.x=t.x+(e.x-t.x)*i,this.y=t.y+(e.y-t.y)*i,this.z=t.z+(e.z-t.z)*i,this.w=t.w+(e.w-t.w)*i,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}},Ba=class extends Jn{constructor(t=1,e=1,i={}){super(),this.isRenderTarget=!0,this.width=t,this.height=e,this.depth=1,this.scissor=new ie(0,0,t,e),this.scissorTest=!1,this.viewport=new ie(0,0,t,e);let s={width:t,height:e,depth:1};i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:hn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1},i);let r=new We(s,i.mapping,i.wrapS,i.wrapT,i.magFilter,i.minFilter,i.format,i.type,i.anisotropy,i.colorSpace);r.flipY=!1,r.generateMipmaps=i.generateMipmaps,r.internalFormat=i.internalFormat,this.textures=[];let o=i.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0;this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this.depthTexture=i.depthTexture,this.samples=i.samples}get texture(){return this.textures[0]}set texture(t){this.textures[0]=t}setSize(t,e,i=1){if(this.width!==t||this.height!==e||this.depth!==i){this.width=t,this.height=e,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=t,this.textures[s].image.height=e,this.textures[s].image.depth=i;this.dispose()}this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.scissor.copy(t.scissor),this.scissorTest=t.scissorTest,this.viewport.copy(t.viewport),this.textures.length=0;for(let i=0,s=t.textures.length;i<s;i++)this.textures[i]=t.textures[i].clone(),this.textures[i].isRenderTargetTexture=!0;let e=Object.assign({},t.texture.image);return this.texture.source=new Ir(e),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.resolveDepthBuffer=t.resolveDepthBuffer,this.resolveStencilBuffer=t.resolveStencilBuffer,t.depthTexture!==null&&(this.depthTexture=t.depthTexture.clone()),this.samples=t.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}},Ln=class extends Ba{constructor(t=1,e=1,i={}){super(t,e,i),this.isWebGLRenderTarget=!0}},Pr=class extends We{constructor(t=null,e=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:i,depth:s},this.magFilter=rn,this.minFilter=rn,this.wrapR=yi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(t){this.layerUpdates.add(t)}clearLayerUpdates(){this.layerUpdates.clear()}};var za=class extends We{constructor(t=null,e=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:i,depth:s},this.magFilter=rn,this.minFilter=rn,this.wrapR=yi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var Kn=class{constructor(t=0,e=0,i=0,s=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=i,this._w=s}static slerpFlat(t,e,i,s,r,o,a){let c=i[s+0],l=i[s+1],h=i[s+2],u=i[s+3],d=r[o+0],f=r[o+1],g=r[o+2],_=r[o+3];if(a===0){t[e+0]=c,t[e+1]=l,t[e+2]=h,t[e+3]=u;return}if(a===1){t[e+0]=d,t[e+1]=f,t[e+2]=g,t[e+3]=_;return}if(u!==_||c!==d||l!==f||h!==g){let m=1-a,p=c*d+l*f+h*g+u*_,S=p>=0?1:-1,M=1-p*p;if(M>Number.EPSILON){let D=Math.sqrt(M),T=Math.atan2(D,p*S);m=Math.sin(m*T)/D,a=Math.sin(a*T)/D}let x=a*S;if(c=c*m+d*x,l=l*m+f*x,h=h*m+g*x,u=u*m+_*x,m===1-a){let D=1/Math.sqrt(c*c+l*l+h*h+u*u);c*=D,l*=D,h*=D,u*=D}}t[e]=c,t[e+1]=l,t[e+2]=h,t[e+3]=u}static multiplyQuaternionsFlat(t,e,i,s,r,o){let a=i[s],c=i[s+1],l=i[s+2],h=i[s+3],u=r[o],d=r[o+1],f=r[o+2],g=r[o+3];return t[e]=a*g+h*u+c*f-l*d,t[e+1]=c*g+h*d+l*u-a*f,t[e+2]=l*g+h*f+a*d-c*u,t[e+3]=h*g-a*u-c*d-l*f,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,i,s){return this._x=t,this._y=e,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e=!0){let i=t._x,s=t._y,r=t._z,o=t._order,a=Math.cos,c=Math.sin,l=a(i/2),h=a(s/2),u=a(r/2),d=c(i/2),f=c(s/2),g=c(r/2);switch(o){case"XYZ":this._x=d*h*u+l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u-d*f*g;break;case"YXZ":this._x=d*h*u+l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u+d*f*g;break;case"ZXY":this._x=d*h*u-l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u-d*f*g;break;case"ZYX":this._x=d*h*u-l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u+d*f*g;break;case"YZX":this._x=d*h*u+l*f*g,this._y=l*f*u+d*h*g,this._z=l*h*g-d*f*u,this._w=l*h*u-d*f*g;break;case"XZY":this._x=d*h*u-l*f*g,this._y=l*f*u-d*h*g,this._z=l*h*g+d*f*u,this._w=l*h*u+d*f*g;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+o)}return e===!0&&this._onChangeCallback(),this}setFromAxisAngle(t,e){let i=e/2,s=Math.sin(i);return this._x=t.x*s,this._y=t.y*s,this._z=t.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(t){let e=t.elements,i=e[0],s=e[4],r=e[8],o=e[1],a=e[5],c=e[9],l=e[2],h=e[6],u=e[10],d=i+a+u;if(d>0){let f=.5/Math.sqrt(d+1);this._w=.25/f,this._x=(h-c)*f,this._y=(r-l)*f,this._z=(o-s)*f}else if(i>a&&i>u){let f=2*Math.sqrt(1+i-a-u);this._w=(h-c)/f,this._x=.25*f,this._y=(s+o)/f,this._z=(r+l)/f}else if(a>u){let f=2*Math.sqrt(1+a-i-u);this._w=(r-l)/f,this._x=(s+o)/f,this._y=.25*f,this._z=(c+h)/f}else{let f=2*Math.sqrt(1+u-i-a);this._w=(o-s)/f,this._x=(r+l)/f,this._y=(c+h)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let i=t.dot(e)+1;return i<Number.EPSILON?(i=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=i):(this._x=0,this._y=-t.z,this._z=t.y,this._w=i)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=i),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(ye(this.dot(t),-1,1)))}rotateTowards(t,e){let i=this.angleTo(t);if(i===0)return this;let s=Math.min(1,e/i);return this.slerp(t,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){let i=t._x,s=t._y,r=t._z,o=t._w,a=e._x,c=e._y,l=e._z,h=e._w;return this._x=i*h+o*a+s*l-r*c,this._y=s*h+o*c+r*a-i*l,this._z=r*h+o*l+i*c-s*a,this._w=o*h-i*a-s*c-r*l,this._onChangeCallback(),this}slerp(t,e){if(e===0)return this;if(e===1)return this.copy(t);let i=this._x,s=this._y,r=this._z,o=this._w,a=o*t._w+i*t._x+s*t._y+r*t._z;if(a<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,a=-a):this.copy(t),a>=1)return this._w=o,this._x=i,this._y=s,this._z=r,this;let c=1-a*a;if(c<=Number.EPSILON){let f=1-e;return this._w=f*o+e*this._w,this._x=f*i+e*this._x,this._y=f*s+e*this._y,this._z=f*r+e*this._z,this.normalize(),this}let l=Math.sqrt(c),h=Math.atan2(l,a),u=Math.sin((1-e)*h)/l,d=Math.sin(e*h)/l;return this._w=o*u+this._w*d,this._x=i*u+this._x*d,this._y=s*u+this._y*d,this._z=r*u+this._z*d,this._onChangeCallback(),this}slerpQuaternions(t,e,i){return this.copy(t).slerp(e,i)}random(){let t=2*Math.PI*Math.random(),e=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(t),s*Math.cos(t),r*Math.sin(e),r*Math.cos(e))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},L=class n{constructor(t=0,e=0,i=0){n.prototype.isVector3=!0,this.x=t,this.y=e,this.z=i}set(t,e,i){return i===void 0&&(i=this.z),this.x=t,this.y=e,this.z=i,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return this.applyQuaternion(bl.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(bl.setFromAxisAngle(t,e))}applyMatrix3(t){let e=this.x,i=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[3]*i+r[6]*s,this.y=r[1]*e+r[4]*i+r[7]*s,this.z=r[2]*e+r[5]*i+r[8]*s,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){let e=this.x,i=this.y,s=this.z,r=t.elements,o=1/(r[3]*e+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*e+r[4]*i+r[8]*s+r[12])*o,this.y=(r[1]*e+r[5]*i+r[9]*s+r[13])*o,this.z=(r[2]*e+r[6]*i+r[10]*s+r[14])*o,this}applyQuaternion(t){let e=this.x,i=this.y,s=this.z,r=t.x,o=t.y,a=t.z,c=t.w,l=2*(o*s-a*i),h=2*(a*e-r*s),u=2*(r*i-o*e);return this.x=e+c*l+o*u-a*h,this.y=i+c*h+a*l-r*u,this.z=s+c*u+r*h-o*l,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){let e=this.x,i=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[4]*i+r[8]*s,this.y=r[1]*e+r[5]*i+r[9]*s,this.z=r[2]*e+r[6]*i+r[10]*s,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this}clampLength(t,e){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(t,Math.min(e,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,i){return this.x=t.x+(e.x-t.x)*i,this.y=t.y+(e.y-t.y)*i,this.z=t.z+(e.z-t.z)*i,this}cross(t){return this.crossVectors(this,t)}crossVectors(t,e){let i=t.x,s=t.y,r=t.z,o=e.x,a=e.y,c=e.z;return this.x=s*c-r*a,this.y=r*o-i*c,this.z=i*a-s*o,this}projectOnVector(t){let e=t.lengthSq();if(e===0)return this.set(0,0,0);let i=t.dot(this)/e;return this.copy(t).multiplyScalar(i)}projectOnPlane(t){return Eo.copy(this).projectOnVector(t),this.sub(Eo)}reflect(t){return this.sub(Eo.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){let e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;let i=this.dot(t)/e;return Math.acos(ye(i,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){let e=this.x-t.x,i=this.y-t.y,s=this.z-t.z;return e*e+i*i+s*s}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,i){let s=Math.sin(e)*t;return this.x=s*Math.sin(i),this.y=Math.cos(e)*t,this.z=s*Math.cos(i),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,i){return this.x=t*Math.sin(e),this.y=i,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){let e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){let e=this.setFromMatrixColumn(t,0).length(),i=this.setFromMatrixColumn(t,1).length(),s=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=i,this.z=s,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,e*4)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,e*3)}setFromEuler(t){return this.x=t._x,this.y=t._y,this.z=t._z,this}setFromColor(t){return this.x=t.r,this.y=t.g,this.z=t.b,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let t=Math.random()*Math.PI*2,e=Math.random()*2-1,i=Math.sqrt(1-e*e);return this.x=i*Math.cos(t),this.y=e,this.z=i*Math.sin(t),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}},Eo=new L,bl=new Kn,Dn=class{constructor(t=new L(1/0,1/0,1/0),e=new L(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,i=t.length;e<i;e+=3)this.expandByPoint(tn.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,i=t.count;e<i;e++)this.expandByPoint(tn.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,i=t.length;e<i;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){let i=tn.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(i),this.max.copy(t).add(i),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){t.updateWorldMatrix(!1,!1);let i=t.geometry;if(i!==void 0){let r=i.getAttribute("position");if(e===!0&&r!==void 0&&t.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)t.isMesh===!0?t.getVertexPosition(o,tn):tn.fromBufferAttribute(r,o),tn.applyMatrix4(t.matrixWorld),this.expandByPoint(tn);else t.boundingBox!==void 0?(t.boundingBox===null&&t.computeBoundingBox(),Qs.copy(t.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Qs.copy(i.boundingBox)),Qs.applyMatrix4(t.matrixWorld),this.union(Qs)}let s=t.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],e);return this}containsPoint(t){return t.x>=this.min.x&&t.x<=this.max.x&&t.y>=this.min.y&&t.y<=this.max.y&&t.z>=this.min.z&&t.z<=this.max.z}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return t.max.x>=this.min.x&&t.min.x<=this.max.x&&t.max.y>=this.min.y&&t.min.y<=this.max.y&&t.max.z>=this.min.z&&t.min.z<=this.max.z}intersectsSphere(t){return this.clampPoint(t.center,tn),tn.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,i;return t.normal.x>0?(e=t.normal.x*this.min.x,i=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,i=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,i+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,i+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,i+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,i+=t.normal.z*this.min.z),e<=-t.constant&&i>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(Ms),js.subVectors(this.max,Ms),Di.subVectors(t.a,Ms),Ui.subVectors(t.b,Ms),Ni.subVectors(t.c,Ms),zn.subVectors(Ui,Di),Hn.subVectors(Ni,Ui),li.subVectors(Di,Ni);let e=[0,-zn.z,zn.y,0,-Hn.z,Hn.y,0,-li.z,li.y,zn.z,0,-zn.x,Hn.z,0,-Hn.x,li.z,0,-li.x,-zn.y,zn.x,0,-Hn.y,Hn.x,0,-li.y,li.x,0];return!bo(e,Di,Ui,Ni,js)||(e=[1,0,0,0,1,0,0,0,1],!bo(e,Di,Ui,Ni,js))?!1:(tr.crossVectors(zn,Hn),e=[tr.x,tr.y,tr.z],bo(e,Di,Ui,Ni,js))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,tn).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(tn).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(Mn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Mn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Mn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Mn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Mn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Mn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Mn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Mn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Mn),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}},Mn=[new L,new L,new L,new L,new L,new L,new L,new L],tn=new L,Qs=new Dn,Di=new L,Ui=new L,Ni=new L,zn=new L,Hn=new L,li=new L,Ms=new L,js=new L,tr=new L,hi=new L;function bo(n,t,e,i,s){for(let r=0,o=n.length-3;r<=o;r+=3){hi.fromArray(n,r);let a=s.x*Math.abs(hi.x)+s.y*Math.abs(hi.y)+s.z*Math.abs(hi.z),c=t.dot(hi),l=e.dot(hi),h=i.dot(hi);if(Math.max(-Math.max(c,l,h),Math.min(c,l,h))>a)return!1}return!0}var Bd=new Dn,Ss=new L,wo=new L,Fs=class{constructor(t=new L,e=-1){this.isSphere=!0,this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){let i=this.center;e!==void 0?i.copy(e):Bd.setFromPoints(t).getCenter(i);let s=0;for(let r=0,o=t.length;r<o;r++)s=Math.max(s,i.distanceToSquared(t[r]));return this.radius=Math.sqrt(s),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){let e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){let i=this.center.distanceToSquared(t);return e.copy(t),i>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;Ss.subVectors(t,this.center);let e=Ss.lengthSq();if(e>this.radius*this.radius){let i=Math.sqrt(e),s=(i-this.radius)*.5;this.center.addScaledVector(Ss,s/i),this.radius+=s}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(wo.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint(Ss.copy(t.center).add(wo)),this.expandByPoint(Ss.copy(t.center).sub(wo))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}},Sn=new L,To=new L,er=new L,kn=new L,Ao=new L,nr=new L,Ro=new L,Lr=class{constructor(t=new L,e=new L(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,Sn)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);let i=e.dot(this.direction);return i<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){let e=Sn.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(Sn.copy(this.origin).addScaledVector(this.direction,e),Sn.distanceToSquared(t))}distanceSqToSegment(t,e,i,s){To.copy(t).add(e).multiplyScalar(.5),er.copy(e).sub(t).normalize(),kn.copy(this.origin).sub(To);let r=t.distanceTo(e)*.5,o=-this.direction.dot(er),a=kn.dot(this.direction),c=-kn.dot(er),l=kn.lengthSq(),h=Math.abs(1-o*o),u,d,f,g;if(h>0)if(u=o*c-a,d=o*a-c,g=r*h,u>=0)if(d>=-g)if(d<=g){let _=1/h;u*=_,d*=_,f=u*(u+o*d+2*a)+d*(o*u+d+2*c)+l}else d=r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;else d=-r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;else d<=-g?(u=Math.max(0,-(-o*r+a)),d=u>0?-r:Math.min(Math.max(-r,-c),r),f=-u*u+d*(d+2*c)+l):d<=g?(u=0,d=Math.min(Math.max(-r,-c),r),f=d*(d+2*c)+l):(u=Math.max(0,-(o*r+a)),d=u>0?r:Math.min(Math.max(-r,-c),r),f=-u*u+d*(d+2*c)+l);else d=o>0?-r:r,u=Math.max(0,-(o*d+a)),f=-u*u+d*(d+2*c)+l;return i&&i.copy(this.origin).addScaledVector(this.direction,u),s&&s.copy(To).addScaledVector(er,d),f}intersectSphere(t,e){Sn.subVectors(t.center,this.origin);let i=Sn.dot(this.direction),s=Sn.dot(Sn)-i*i,r=t.radius*t.radius;if(s>r)return null;let o=Math.sqrt(r-s),a=i-o,c=i+o;return c<0?null:a<0?this.at(c,e):this.at(a,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){let e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(t.normal)+t.constant)/e;return i>=0?i:null}intersectPlane(t,e){let i=this.distanceToPlane(t);return i===null?null:this.at(i,e)}intersectsPlane(t){let e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let i,s,r,o,a,c,l=1/this.direction.x,h=1/this.direction.y,u=1/this.direction.z,d=this.origin;return l>=0?(i=(t.min.x-d.x)*l,s=(t.max.x-d.x)*l):(i=(t.max.x-d.x)*l,s=(t.min.x-d.x)*l),h>=0?(r=(t.min.y-d.y)*h,o=(t.max.y-d.y)*h):(r=(t.max.y-d.y)*h,o=(t.min.y-d.y)*h),i>o||r>s||((r>i||isNaN(i))&&(i=r),(o<s||isNaN(s))&&(s=o),u>=0?(a=(t.min.z-d.z)*u,c=(t.max.z-d.z)*u):(a=(t.max.z-d.z)*u,c=(t.min.z-d.z)*u),i>c||a>s)||((a>i||i!==i)&&(i=a),(c<s||s!==s)&&(s=c),s<0)?null:this.at(i>=0?i:s,e)}intersectsBox(t){return this.intersectBox(t,Sn)!==null}intersectTriangle(t,e,i,s,r){Ao.subVectors(e,t),nr.subVectors(i,t),Ro.crossVectors(Ao,nr);let o=this.direction.dot(Ro),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;kn.subVectors(this.origin,t);let c=a*this.direction.dot(nr.crossVectors(kn,nr));if(c<0)return null;let l=a*this.direction.dot(Ao.cross(kn));if(l<0||c+l>o)return null;let h=-a*kn.dot(Ro);return h<0?null:this.at(h/o,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},de=class n{constructor(t,e,i,s,r,o,a,c,l,h,u,d,f,g,_,m){n.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],t!==void 0&&this.set(t,e,i,s,r,o,a,c,l,h,u,d,f,g,_,m)}set(t,e,i,s,r,o,a,c,l,h,u,d,f,g,_,m){let p=this.elements;return p[0]=t,p[4]=e,p[8]=i,p[12]=s,p[1]=r,p[5]=o,p[9]=a,p[13]=c,p[2]=l,p[6]=h,p[10]=u,p[14]=d,p[3]=f,p[7]=g,p[11]=_,p[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new n().fromArray(this.elements)}copy(t){let e=this.elements,i=t.elements;return e[0]=i[0],e[1]=i[1],e[2]=i[2],e[3]=i[3],e[4]=i[4],e[5]=i[5],e[6]=i[6],e[7]=i[7],e[8]=i[8],e[9]=i[9],e[10]=i[10],e[11]=i[11],e[12]=i[12],e[13]=i[13],e[14]=i[14],e[15]=i[15],this}copyPosition(t){let e=this.elements,i=t.elements;return e[12]=i[12],e[13]=i[13],e[14]=i[14],this}setFromMatrix3(t){let e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,i){return t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this}makeBasis(t,e,i){return this.set(t.x,e.x,i.x,0,t.y,e.y,i.y,0,t.z,e.z,i.z,0,0,0,0,1),this}extractRotation(t){let e=this.elements,i=t.elements,s=1/Fi.setFromMatrixColumn(t,0).length(),r=1/Fi.setFromMatrixColumn(t,1).length(),o=1/Fi.setFromMatrixColumn(t,2).length();return e[0]=i[0]*s,e[1]=i[1]*s,e[2]=i[2]*s,e[3]=0,e[4]=i[4]*r,e[5]=i[5]*r,e[6]=i[6]*r,e[7]=0,e[8]=i[8]*o,e[9]=i[9]*o,e[10]=i[10]*o,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){let e=this.elements,i=t.x,s=t.y,r=t.z,o=Math.cos(i),a=Math.sin(i),c=Math.cos(s),l=Math.sin(s),h=Math.cos(r),u=Math.sin(r);if(t.order==="XYZ"){let d=o*h,f=o*u,g=a*h,_=a*u;e[0]=c*h,e[4]=-c*u,e[8]=l,e[1]=f+g*l,e[5]=d-_*l,e[9]=-a*c,e[2]=_-d*l,e[6]=g+f*l,e[10]=o*c}else if(t.order==="YXZ"){let d=c*h,f=c*u,g=l*h,_=l*u;e[0]=d+_*a,e[4]=g*a-f,e[8]=o*l,e[1]=o*u,e[5]=o*h,e[9]=-a,e[2]=f*a-g,e[6]=_+d*a,e[10]=o*c}else if(t.order==="ZXY"){let d=c*h,f=c*u,g=l*h,_=l*u;e[0]=d-_*a,e[4]=-o*u,e[8]=g+f*a,e[1]=f+g*a,e[5]=o*h,e[9]=_-d*a,e[2]=-o*l,e[6]=a,e[10]=o*c}else if(t.order==="ZYX"){let d=o*h,f=o*u,g=a*h,_=a*u;e[0]=c*h,e[4]=g*l-f,e[8]=d*l+_,e[1]=c*u,e[5]=_*l+d,e[9]=f*l-g,e[2]=-l,e[6]=a*c,e[10]=o*c}else if(t.order==="YZX"){let d=o*c,f=o*l,g=a*c,_=a*l;e[0]=c*h,e[4]=_-d*u,e[8]=g*u+f,e[1]=u,e[5]=o*h,e[9]=-a*h,e[2]=-l*h,e[6]=f*u+g,e[10]=d-_*u}else if(t.order==="XZY"){let d=o*c,f=o*l,g=a*c,_=a*l;e[0]=c*h,e[4]=-u,e[8]=l*h,e[1]=d*u+_,e[5]=o*h,e[9]=f*u-g,e[2]=g*u-f,e[6]=a*h,e[10]=_*u+d}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(zd,t,Hd)}lookAt(t,e,i){let s=this.elements;return Ve.subVectors(t,e),Ve.lengthSq()===0&&(Ve.z=1),Ve.normalize(),Vn.crossVectors(i,Ve),Vn.lengthSq()===0&&(Math.abs(i.z)===1?Ve.x+=1e-4:Ve.z+=1e-4,Ve.normalize(),Vn.crossVectors(i,Ve)),Vn.normalize(),ir.crossVectors(Ve,Vn),s[0]=Vn.x,s[4]=ir.x,s[8]=Ve.x,s[1]=Vn.y,s[5]=ir.y,s[9]=Ve.y,s[2]=Vn.z,s[6]=ir.z,s[10]=Ve.z,this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){let i=t.elements,s=e.elements,r=this.elements,o=i[0],a=i[4],c=i[8],l=i[12],h=i[1],u=i[5],d=i[9],f=i[13],g=i[2],_=i[6],m=i[10],p=i[14],S=i[3],M=i[7],x=i[11],D=i[15],T=s[0],R=s[4],C=s[8],b=s[12],y=s[1],I=s[5],B=s[9],N=s[13],H=s[2],Y=s[6],V=s[10],rt=s[14],X=s[3],it=s[7],dt=s[11],vt=s[15];return r[0]=o*T+a*y+c*H+l*X,r[4]=o*R+a*I+c*Y+l*it,r[8]=o*C+a*B+c*V+l*dt,r[12]=o*b+a*N+c*rt+l*vt,r[1]=h*T+u*y+d*H+f*X,r[5]=h*R+u*I+d*Y+f*it,r[9]=h*C+u*B+d*V+f*dt,r[13]=h*b+u*N+d*rt+f*vt,r[2]=g*T+_*y+m*H+p*X,r[6]=g*R+_*I+m*Y+p*it,r[10]=g*C+_*B+m*V+p*dt,r[14]=g*b+_*N+m*rt+p*vt,r[3]=S*T+M*y+x*H+D*X,r[7]=S*R+M*I+x*Y+D*it,r[11]=S*C+M*B+x*V+D*dt,r[15]=S*b+M*N+x*rt+D*vt,this}multiplyScalar(t){let e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){let t=this.elements,e=t[0],i=t[4],s=t[8],r=t[12],o=t[1],a=t[5],c=t[9],l=t[13],h=t[2],u=t[6],d=t[10],f=t[14],g=t[3],_=t[7],m=t[11],p=t[15];return g*(+r*c*u-s*l*u-r*a*d+i*l*d+s*a*f-i*c*f)+_*(+e*c*f-e*l*d+r*o*d-s*o*f+s*l*h-r*c*h)+m*(+e*l*u-e*a*f-r*o*u+i*o*f+r*a*h-i*l*h)+p*(-s*a*h-e*c*u+e*a*d+s*o*u-i*o*d+i*c*h)}transpose(){let t=this.elements,e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,i){let s=this.elements;return t.isVector3?(s[12]=t.x,s[13]=t.y,s[14]=t.z):(s[12]=t,s[13]=e,s[14]=i),this}invert(){let t=this.elements,e=t[0],i=t[1],s=t[2],r=t[3],o=t[4],a=t[5],c=t[6],l=t[7],h=t[8],u=t[9],d=t[10],f=t[11],g=t[12],_=t[13],m=t[14],p=t[15],S=u*m*l-_*d*l+_*c*f-a*m*f-u*c*p+a*d*p,M=g*d*l-h*m*l-g*c*f+o*m*f+h*c*p-o*d*p,x=h*_*l-g*u*l+g*a*f-o*_*f-h*a*p+o*u*p,D=g*u*c-h*_*c-g*a*d+o*_*d+h*a*m-o*u*m,T=e*S+i*M+s*x+r*D;if(T===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let R=1/T;return t[0]=S*R,t[1]=(_*d*r-u*m*r-_*s*f+i*m*f+u*s*p-i*d*p)*R,t[2]=(a*m*r-_*c*r+_*s*l-i*m*l-a*s*p+i*c*p)*R,t[3]=(u*c*r-a*d*r-u*s*l+i*d*l+a*s*f-i*c*f)*R,t[4]=M*R,t[5]=(h*m*r-g*d*r+g*s*f-e*m*f-h*s*p+e*d*p)*R,t[6]=(g*c*r-o*m*r-g*s*l+e*m*l+o*s*p-e*c*p)*R,t[7]=(o*d*r-h*c*r+h*s*l-e*d*l-o*s*f+e*c*f)*R,t[8]=x*R,t[9]=(g*u*r-h*_*r-g*i*f+e*_*f+h*i*p-e*u*p)*R,t[10]=(o*_*r-g*a*r+g*i*l-e*_*l-o*i*p+e*a*p)*R,t[11]=(h*a*r-o*u*r-h*i*l+e*u*l+o*i*f-e*a*f)*R,t[12]=D*R,t[13]=(h*_*s-g*u*s+g*i*d-e*_*d-h*i*m+e*u*m)*R,t[14]=(g*a*s-o*_*s-g*i*c+e*_*c+o*i*m-e*a*m)*R,t[15]=(o*u*s-h*a*s+h*i*c-e*u*c-o*i*d+e*a*d)*R,this}scale(t){let e=this.elements,i=t.x,s=t.y,r=t.z;return e[0]*=i,e[4]*=s,e[8]*=r,e[1]*=i,e[5]*=s,e[9]*=r,e[2]*=i,e[6]*=s,e[10]*=r,e[3]*=i,e[7]*=s,e[11]*=r,this}getMaxScaleOnAxis(){let t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],i=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],s=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,i,s))}makeTranslation(t,e,i){return t.isVector3?this.set(1,0,0,t.x,0,1,0,t.y,0,0,1,t.z,0,0,0,1):this.set(1,0,0,t,0,1,0,e,0,0,1,i,0,0,0,1),this}makeRotationX(t){let e=Math.cos(t),i=Math.sin(t);return this.set(1,0,0,0,0,e,-i,0,0,i,e,0,0,0,0,1),this}makeRotationY(t){let e=Math.cos(t),i=Math.sin(t);return this.set(e,0,i,0,0,1,0,0,-i,0,e,0,0,0,0,1),this}makeRotationZ(t){let e=Math.cos(t),i=Math.sin(t);return this.set(e,-i,0,0,i,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){let i=Math.cos(e),s=Math.sin(e),r=1-i,o=t.x,a=t.y,c=t.z,l=r*o,h=r*a;return this.set(l*o+i,l*a-s*c,l*c+s*a,0,l*a+s*c,h*a+i,h*c-s*o,0,l*c-s*a,h*c+s*o,r*c*c+i,0,0,0,0,1),this}makeScale(t,e,i){return this.set(t,0,0,0,0,e,0,0,0,0,i,0,0,0,0,1),this}makeShear(t,e,i,s,r,o){return this.set(1,i,r,0,t,1,o,0,e,s,1,0,0,0,0,1),this}compose(t,e,i){let s=this.elements,r=e._x,o=e._y,a=e._z,c=e._w,l=r+r,h=o+o,u=a+a,d=r*l,f=r*h,g=r*u,_=o*h,m=o*u,p=a*u,S=c*l,M=c*h,x=c*u,D=i.x,T=i.y,R=i.z;return s[0]=(1-(_+p))*D,s[1]=(f+x)*D,s[2]=(g-M)*D,s[3]=0,s[4]=(f-x)*T,s[5]=(1-(d+p))*T,s[6]=(m+S)*T,s[7]=0,s[8]=(g+M)*R,s[9]=(m-S)*R,s[10]=(1-(d+_))*R,s[11]=0,s[12]=t.x,s[13]=t.y,s[14]=t.z,s[15]=1,this}decompose(t,e,i){let s=this.elements,r=Fi.set(s[0],s[1],s[2]).length(),o=Fi.set(s[4],s[5],s[6]).length(),a=Fi.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),t.x=s[12],t.y=s[13],t.z=s[14],en.copy(this);let l=1/r,h=1/o,u=1/a;return en.elements[0]*=l,en.elements[1]*=l,en.elements[2]*=l,en.elements[4]*=h,en.elements[5]*=h,en.elements[6]*=h,en.elements[8]*=u,en.elements[9]*=u,en.elements[10]*=u,e.setFromRotationMatrix(en),i.x=r,i.y=o,i.z=a,this}makePerspective(t,e,i,s,r,o,a=Cn){let c=this.elements,l=2*r/(e-t),h=2*r/(i-s),u=(e+t)/(e-t),d=(i+s)/(i-s),f,g;if(a===Cn)f=-(o+r)/(o-r),g=-2*o*r/(o-r);else if(a===Rr)f=-o/(o-r),g=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=l,c[4]=0,c[8]=u,c[12]=0,c[1]=0,c[5]=h,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=f,c[14]=g,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(t,e,i,s,r,o,a=Cn){let c=this.elements,l=1/(e-t),h=1/(i-s),u=1/(o-r),d=(e+t)*l,f=(i+s)*h,g,_;if(a===Cn)g=(o+r)*u,_=-2*u;else if(a===Rr)g=r*u,_=-1*u;else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=2*l,c[4]=0,c[8]=0,c[12]=-d,c[1]=0,c[5]=2*h,c[9]=0,c[13]=-f,c[2]=0,c[6]=0,c[10]=_,c[14]=-g,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(t){let e=this.elements,i=t.elements;for(let s=0;s<16;s++)if(e[s]!==i[s])return!1;return!0}fromArray(t,e=0){for(let i=0;i<16;i++)this.elements[i]=t[i+e];return this}toArray(t=[],e=0){let i=this.elements;return t[e]=i[0],t[e+1]=i[1],t[e+2]=i[2],t[e+3]=i[3],t[e+4]=i[4],t[e+5]=i[5],t[e+6]=i[6],t[e+7]=i[7],t[e+8]=i[8],t[e+9]=i[9],t[e+10]=i[10],t[e+11]=i[11],t[e+12]=i[12],t[e+13]=i[13],t[e+14]=i[14],t[e+15]=i[15],t}},Fi=new L,en=new de,zd=new L(0,0,0),Hd=new L(1,1,1),Vn=new L,ir=new L,Ve=new L,wl=new de,Tl=new Kn,un=class n{constructor(t=0,e=0,i=0,s=n.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=e,this._z=i,this._order=s}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,i,s=this._order){return this._x=t,this._y=e,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,i=!0){let s=t.elements,r=s[0],o=s[4],a=s[8],c=s[1],l=s[5],h=s[9],u=s[2],d=s[6],f=s[10];switch(e){case"XYZ":this._y=Math.asin(ye(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-h,f),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,l),this._z=0);break;case"YXZ":this._x=Math.asin(-ye(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(a,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-u,r),this._z=0);break;case"ZXY":this._x=Math.asin(ye(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-u,f),this._z=Math.atan2(-o,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-ye(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(d,f),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-o,l));break;case"YZX":this._z=Math.asin(ye(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-u,r)):(this._x=0,this._y=Math.atan2(a,f));break;case"XZY":this._z=Math.asin(-ye(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,l),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-h,f),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,i===!0&&this._onChangeCallback(),this}setFromQuaternion(t,e,i){return wl.makeRotationFromQuaternion(t),this.setFromRotationMatrix(wl,e,i)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return Tl.setFromEuler(this),this.setFromQuaternion(Tl,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],t[3]!==void 0&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};un.DEFAULT_ORDER="XYZ";var Os=class{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}},kd=0,Al=new L,Oi=new Kn,En=new de,sr=new L,Es=new L,Vd=new L,Gd=new Kn,Rl=new L(1,0,0),Cl=new L(0,1,0),Il=new L(0,0,1),Pl={type:"added"},Wd={type:"removed"},Bi={type:"childadded",child:null},Co={type:"childremoved",child:null},Ue=class n extends Jn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:kd++}),this.uuid=Ri(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=n.DEFAULT_UP.clone();let t=new L,e=new un,i=new Kn,s=new L(1,1,1);function r(){i.setFromEuler(e,!1)}function o(){e.setFromQuaternion(i,void 0,!1)}e._onChange(r),i._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new de},normalMatrix:{value:new Wt}}),this.matrix=new de,this.matrixWorld=new de,this.matrixAutoUpdate=n.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=n.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Os,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return Oi.setFromAxisAngle(t,e),this.quaternion.multiply(Oi),this}rotateOnWorldAxis(t,e){return Oi.setFromAxisAngle(t,e),this.quaternion.premultiply(Oi),this}rotateX(t){return this.rotateOnAxis(Rl,t)}rotateY(t){return this.rotateOnAxis(Cl,t)}rotateZ(t){return this.rotateOnAxis(Il,t)}translateOnAxis(t,e){return Al.copy(t).applyQuaternion(this.quaternion),this.position.add(Al.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(Rl,t)}translateY(t){return this.translateOnAxis(Cl,t)}translateZ(t){return this.translateOnAxis(Il,t)}localToWorld(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(En.copy(this.matrixWorld).invert())}lookAt(t,e,i){t.isVector3?sr.copy(t):sr.set(t,e,i);let s=this.parent;this.updateWorldMatrix(!0,!1),Es.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?En.lookAt(Es,sr,this.up):En.lookAt(sr,Es,this.up),this.quaternion.setFromRotationMatrix(En),s&&(En.extractRotation(s.matrixWorld),Oi.setFromRotationMatrix(En),this.quaternion.premultiply(Oi.invert()))}add(t){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return t===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(t.removeFromParent(),t.parent=this,this.children.push(t),t.dispatchEvent(Pl),Bi.child=t,this.dispatchEvent(Bi),Bi.child=null):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let e=this.children.indexOf(t);return e!==-1&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(Wd),Co.child=t,this.dispatchEvent(Co),Co.child=null),this}removeFromParent(){let t=this.parent;return t!==null&&t.remove(this),this}clear(){return this.remove(...this.children)}attach(t){return this.updateWorldMatrix(!0,!1),En.copy(this.matrixWorld).invert(),t.parent!==null&&(t.parent.updateWorldMatrix(!0,!1),En.multiply(t.parent.matrixWorld)),t.applyMatrix4(En),t.removeFromParent(),t.parent=this,this.children.push(t),t.updateWorldMatrix(!1,!0),t.dispatchEvent(Pl),Bi.child=t,this.dispatchEvent(Bi),Bi.child=null,this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let i=0,s=this.children.length;i<s;i++){let o=this.children[i].getObjectByProperty(t,e);if(o!==void 0)return o}}getObjectsByProperty(t,e,i=[]){this[t]===e&&i.push(this);let s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(t,e,i);return i}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Es,t,Vd),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Es,Gd,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);let e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(t){t(this);let e=this.children;for(let i=0,s=e.length;i<s;i++)e[i].traverse(t)}traverseVisible(t){if(this.visible===!1)return;t(this);let e=this.children;for(let i=0,s=e.length;i<s;i++)e[i].traverseVisible(t)}traverseAncestors(t){let e=this.parent;e!==null&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,t=!0);let e=this.children;for(let i=0,s=e.length;i<s;i++)e[i].updateMatrixWorld(t)}updateWorldMatrix(t,e){let i=this.parent;if(t===!0&&i!==null&&i.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),e===!0){let s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(t){let e=t===void 0||typeof t=="string",i={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.6,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.visibility=this._visibility,s.active=this._active,s.bounds=this._bounds.map(a=>({boxInitialized:a.boxInitialized,boxMin:a.box.min.toArray(),boxMax:a.box.max.toArray(),sphereInitialized:a.sphereInitialized,sphereRadius:a.sphere.radius,sphereCenter:a.sphere.center.toArray()})),s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.geometryCount=this._geometryCount,s.matricesTexture=this._matricesTexture.toJSON(t),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(t)),this.boundingSphere!==null&&(s.boundingSphere={center:s.boundingSphere.center.toArray(),radius:s.boundingSphere.radius}),this.boundingBox!==null&&(s.boundingBox={min:s.boundingBox.min.toArray(),max:s.boundingBox.max.toArray()}));function r(a,c){return a[c.uuid]===void 0&&(a[c.uuid]=c.toJSON(t)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(t.geometries,this.geometry);let a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){let c=a.shapes;if(Array.isArray(c))for(let l=0,h=c.length;l<h;l++){let u=c[l];r(t.shapes,u)}else r(t.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(t.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let a=[];for(let c=0,l=this.material.length;c<l;c++)a.push(r(t.materials,this.material[c]));s.material=a}else s.material=r(t.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(t).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){let c=this.animations[a];s.animations.push(r(t.animations,c))}}if(e){let a=o(t.geometries),c=o(t.materials),l=o(t.textures),h=o(t.images),u=o(t.shapes),d=o(t.skeletons),f=o(t.animations),g=o(t.nodes);a.length>0&&(i.geometries=a),c.length>0&&(i.materials=c),l.length>0&&(i.textures=l),h.length>0&&(i.images=h),u.length>0&&(i.shapes=u),d.length>0&&(i.skeletons=d),f.length>0&&(i.animations=f),g.length>0&&(i.nodes=g)}return i.object=s,i;function o(a){let c=[];for(let l in a){let h=a[l];delete h.metadata,c.push(h)}return c}}clone(t){return new this.constructor().copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldAutoUpdate=t.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.animations=t.animations.slice(),this.userData=JSON.parse(JSON.stringify(t.userData)),e===!0)for(let i=0;i<t.children.length;i++){let s=t.children[i];this.add(s.clone())}return this}};Ue.DEFAULT_UP=new L(0,1,0);Ue.DEFAULT_MATRIX_AUTO_UPDATE=!0;Ue.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var nn=new L,bn=new L,Io=new L,wn=new L,zi=new L,Hi=new L,Ll=new L,Po=new L,Lo=new L,Do=new L,Uo=new ie,No=new ie,Fo=new ie,_i=class n{constructor(t=new L,e=new L,i=new L){this.a=t,this.b=e,this.c=i}static getNormal(t,e,i,s){s.subVectors(i,e),nn.subVectors(t,e),s.cross(nn);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(t,e,i,s,r){nn.subVectors(s,e),bn.subVectors(i,e),Io.subVectors(t,e);let o=nn.dot(nn),a=nn.dot(bn),c=nn.dot(Io),l=bn.dot(bn),h=bn.dot(Io),u=o*l-a*a;if(u===0)return r.set(0,0,0),null;let d=1/u,f=(l*c-a*h)*d,g=(o*h-a*c)*d;return r.set(1-f-g,g,f)}static containsPoint(t,e,i,s){return this.getBarycoord(t,e,i,s,wn)===null?!1:wn.x>=0&&wn.y>=0&&wn.x+wn.y<=1}static getInterpolation(t,e,i,s,r,o,a,c){return this.getBarycoord(t,e,i,s,wn)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,wn.x),c.addScaledVector(o,wn.y),c.addScaledVector(a,wn.z),c)}static getInterpolatedAttribute(t,e,i,s,r,o){return Uo.setScalar(0),No.setScalar(0),Fo.setScalar(0),Uo.fromBufferAttribute(t,e),No.fromBufferAttribute(t,i),Fo.fromBufferAttribute(t,s),o.setScalar(0),o.addScaledVector(Uo,r.x),o.addScaledVector(No,r.y),o.addScaledVector(Fo,r.z),o}static isFrontFacing(t,e,i,s){return nn.subVectors(i,e),bn.subVectors(t,e),nn.cross(bn).dot(s)<0}set(t,e,i){return this.a.copy(t),this.b.copy(e),this.c.copy(i),this}setFromPointsAndIndices(t,e,i,s){return this.a.copy(t[e]),this.b.copy(t[i]),this.c.copy(t[s]),this}setFromAttributeAndIndices(t,e,i,s){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,i),this.c.fromBufferAttribute(t,s),this}clone(){return new this.constructor().copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return nn.subVectors(this.c,this.b),bn.subVectors(this.a,this.b),nn.cross(bn).length()*.5}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return n.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return n.getBarycoord(t,this.a,this.b,this.c,e)}getInterpolation(t,e,i,s,r){return n.getInterpolation(t,this.a,this.b,this.c,e,i,s,r)}containsPoint(t){return n.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return n.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){let i=this.a,s=this.b,r=this.c,o,a;zi.subVectors(s,i),Hi.subVectors(r,i),Po.subVectors(t,i);let c=zi.dot(Po),l=Hi.dot(Po);if(c<=0&&l<=0)return e.copy(i);Lo.subVectors(t,s);let h=zi.dot(Lo),u=Hi.dot(Lo);if(h>=0&&u<=h)return e.copy(s);let d=c*u-h*l;if(d<=0&&c>=0&&h<=0)return o=c/(c-h),e.copy(i).addScaledVector(zi,o);Do.subVectors(t,r);let f=zi.dot(Do),g=Hi.dot(Do);if(g>=0&&f<=g)return e.copy(r);let _=f*l-c*g;if(_<=0&&l>=0&&g<=0)return a=l/(l-g),e.copy(i).addScaledVector(Hi,a);let m=h*g-f*u;if(m<=0&&u-h>=0&&f-g>=0)return Ll.subVectors(r,s),a=(u-h)/(u-h+(f-g)),e.copy(s).addScaledVector(Ll,a);let p=1/(m+_+d);return o=_*p,a=d*p,e.copy(i).addScaledVector(zi,o).addScaledVector(Hi,a)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}},Uh={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Gn={h:0,s:0,l:0},rr={h:0,s:0,l:0};function Oo(n,t,e){return e<0&&(e+=1),e>1&&(e-=1),e<1/6?n+(t-n)*6*e:e<1/2?t:e<2/3?n+(t-n)*6*(2/3-e):n}var kt=class{constructor(t,e,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(t,e,i)}set(t,e,i){if(e===void 0&&i===void 0){let s=t;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(t,e,i);return this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=xe){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,Jt.toWorkingColorSpace(this,e),this}setRGB(t,e,i,s=Jt.workingColorSpace){return this.r=t,this.g=e,this.b=i,Jt.toWorkingColorSpace(this,s),this}setHSL(t,e,i,s=Jt.workingColorSpace){if(t=Lc(t,1),e=ye(e,0,1),i=ye(i,0,1),e===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+e):i+e-i*e,o=2*i-r;this.r=Oo(o,r,t+1/3),this.g=Oo(o,r,t),this.b=Oo(o,r,t-1/3)}return Jt.toWorkingColorSpace(this,s),this}setStyle(t,e=xe){function i(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(t)){let r,o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,e);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,e);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,e);break;default:console.warn("THREE.Color: Unknown color model "+t)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(t)){let r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(o===6)return this.setHex(parseInt(r,16),e);console.warn("THREE.Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=xe){let i=Uh[t.toLowerCase()];return i!==void 0?this.setHex(i,e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=In(t.r),this.g=In(t.g),this.b=In(t.b),this}copyLinearToSRGB(t){return this.r=Ji(t.r),this.g=Ji(t.g),this.b=Ji(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=xe){return Jt.fromWorkingColorSpace(Ie.copy(this),t),Math.round(ye(Ie.r*255,0,255))*65536+Math.round(ye(Ie.g*255,0,255))*256+Math.round(ye(Ie.b*255,0,255))}getHexString(t=xe){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=Jt.workingColorSpace){Jt.fromWorkingColorSpace(Ie.copy(this),e);let i=Ie.r,s=Ie.g,r=Ie.b,o=Math.max(i,s,r),a=Math.min(i,s,r),c,l,h=(a+o)/2;if(a===o)c=0,l=0;else{let u=o-a;switch(l=h<=.5?u/(o+a):u/(2-o-a),o){case i:c=(s-r)/u+(s<r?6:0);break;case s:c=(r-i)/u+2;break;case r:c=(i-s)/u+4;break}c/=6}return t.h=c,t.s=l,t.l=h,t}getRGB(t,e=Jt.workingColorSpace){return Jt.fromWorkingColorSpace(Ie.copy(this),e),t.r=Ie.r,t.g=Ie.g,t.b=Ie.b,t}getStyle(t=xe){Jt.fromWorkingColorSpace(Ie.copy(this),t);let e=Ie.r,i=Ie.g,s=Ie.b;return t!==xe?`color(${t} ${e.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(e*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(t,e,i){return this.getHSL(Gn),this.setHSL(Gn.h+t,Gn.s+e,Gn.l+i)}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,i){return this.r=t.r+(e.r-t.r)*i,this.g=t.g+(e.g-t.g)*i,this.b=t.b+(e.b-t.b)*i,this}lerpHSL(t,e){this.getHSL(Gn),t.getHSL(rr);let i=Cs(Gn.h,rr.h,e),s=Cs(Gn.s,rr.s,e),r=Cs(Gn.l,rr.l,e);return this.setHSL(i,s,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){let e=this.r,i=this.g,s=this.b,r=t.elements;return this.r=r[0]*e+r[3]*i+r[6]*s,this.g=r[1]*e+r[4]*i+r[7]*s,this.b=r[2]*e+r[5]*i+r[8]*s,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},Ie=new kt;kt.NAMES=Uh;var Xd=0,Si=class extends Jn{static get type(){return"Material"}get type(){return this.constructor.type}set type(t){}constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Xd++}),this.uuid=Ri(),this.name="",this.blending=Zi,this.side=$n,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Qo,this.blendDst=jo,this.blendEquation=gi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new kt(0,0,0),this.blendAlpha=0,this.depthFunc=Qi,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=pl,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Pi,this.stencilZFail=Pi,this.stencilZPass=Pi,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(let e in t){let i=t[e];if(i===void 0){console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);continue}let s=this[e];if(s===void 0){console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[e]=i}}toJSON(t){let e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});let i={metadata:{version:4.6,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(t).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(t).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(t).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(t).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(t).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==Zi&&(i.blending=this.blending),this.side!==$n&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==Qo&&(i.blendSrc=this.blendSrc),this.blendDst!==jo&&(i.blendDst=this.blendDst),this.blendEquation!==gi&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==Qi&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==pl&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Pi&&(i.stencilFail=this.stencilFail),this.stencilZFail!==Pi&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==Pi&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let o=[];for(let a in r){let c=r[a];delete c.metadata,o.push(c)}return o}if(e){let r=s(t.textures),o=s(t.images);r.length>0&&(i.textures=r),o.length>0&&(i.images=o)}return i}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.blendColor.copy(t.blendColor),this.blendAlpha=t.blendAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;let e=t.clippingPlanes,i=null;if(e!==null){let s=e.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=e[r].clone()}return this.clippingPlanes=i,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaHash=t.alphaHash,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}onBuild(){console.warn("Material: onBuild() has been removed.")}},Dr=class extends Si{static get type(){return"MeshBasicMaterial"}constructor(t){super(),this.isMeshBasicMaterial=!0,this.color=new kt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new un,this.combine=yh,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}};var _e=new L,or=new j,we=class{constructor(t,e,i=!1){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=i,this.usage=ml,this.updateRanges=[],this.gpuType=Rn,this.version=0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this.gpuType=t.gpuType,this}copyAt(t,e,i){t*=this.itemSize,i*=e.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[t+s]=e.array[i+s];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,i=this.count;e<i;e++)or.fromBufferAttribute(this,e),or.applyMatrix3(t),this.setXY(e,or.x,or.y);else if(this.itemSize===3)for(let e=0,i=this.count;e<i;e++)_e.fromBufferAttribute(this,e),_e.applyMatrix3(t),this.setXYZ(e,_e.x,_e.y,_e.z);return this}applyMatrix4(t){for(let e=0,i=this.count;e<i;e++)_e.fromBufferAttribute(this,e),_e.applyMatrix4(t),this.setXYZ(e,_e.x,_e.y,_e.z);return this}applyNormalMatrix(t){for(let e=0,i=this.count;e<i;e++)_e.fromBufferAttribute(this,e),_e.applyNormalMatrix(t),this.setXYZ(e,_e.x,_e.y,_e.z);return this}transformDirection(t){for(let e=0,i=this.count;e<i;e++)_e.fromBufferAttribute(this,e),_e.transformDirection(t),this.setXYZ(e,_e.x,_e.y,_e.z);return this}set(t,e=0){return this.array.set(t,e),this}getComponent(t,e){let i=this.array[t*this.itemSize+e];return this.normalized&&(i=Xi(i,this.array)),i}setComponent(t,e,i){return this.normalized&&(i=Le(i,this.array)),this.array[t*this.itemSize+e]=i,this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=Xi(e,this.array)),e}setX(t,e){return this.normalized&&(e=Le(e,this.array)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=Xi(e,this.array)),e}setY(t,e){return this.normalized&&(e=Le(e,this.array)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=Xi(e,this.array)),e}setZ(t,e){return this.normalized&&(e=Le(e,this.array)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=Xi(e,this.array)),e}setW(t,e){return this.normalized&&(e=Le(e,this.array)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,i){return t*=this.itemSize,this.normalized&&(e=Le(e,this.array),i=Le(i,this.array)),this.array[t+0]=e,this.array[t+1]=i,this}setXYZ(t,e,i,s){return t*=this.itemSize,this.normalized&&(e=Le(e,this.array),i=Le(i,this.array),s=Le(s,this.array)),this.array[t+0]=e,this.array[t+1]=i,this.array[t+2]=s,this}setXYZW(t,e,i,s,r){return t*=this.itemSize,this.normalized&&(e=Le(e,this.array),i=Le(i,this.array),s=Le(s,this.array),r=Le(r,this.array)),this.array[t+0]=e,this.array[t+1]=i,this.array[t+2]=s,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(t.name=this.name),this.usage!==ml&&(t.usage=this.usage),t}};var Ur=class extends we{constructor(t,e,i){super(new Uint16Array(t),e,i)}};var Nr=class extends we{constructor(t,e,i){super(new Uint32Array(t),e,i)}};var se=class extends we{constructor(t,e,i){super(new Float32Array(t),e,i)}},qd=0,Ze=new de,Bo=new Ue,ki=new L,Ge=new Dn,bs=new Dn,Ee=new L,Te=class n extends Jn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:qd++}),this.uuid=Ri(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(Dh(t)?Nr:Ur)(t,1):this.index=t,this}setIndirect(t){return this.indirect=t,this}getIndirect(){return this.indirect}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return this.attributes[t]!==void 0}addGroup(t,e,i=0){this.groups.push({start:t,count:e,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){let e=this.attributes.position;e!==void 0&&(e.applyMatrix4(t),e.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new Wt().getNormalMatrix(t);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(t),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(t){return Ze.makeRotationFromQuaternion(t),this.applyMatrix4(Ze),this}rotateX(t){return Ze.makeRotationX(t),this.applyMatrix4(Ze),this}rotateY(t){return Ze.makeRotationY(t),this.applyMatrix4(Ze),this}rotateZ(t){return Ze.makeRotationZ(t),this.applyMatrix4(Ze),this}translate(t,e,i){return Ze.makeTranslation(t,e,i),this.applyMatrix4(Ze),this}scale(t,e,i){return Ze.makeScale(t,e,i),this.applyMatrix4(Ze),this}lookAt(t){return Bo.lookAt(t),Bo.updateMatrix(),this.applyMatrix4(Bo.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(ki).negate(),this.translate(ki.x,ki.y,ki.z),this}setFromPoints(t){let e=this.getAttribute("position");if(e===void 0){let i=[];for(let s=0,r=t.length;s<r;s++){let o=t[s];i.push(o.x,o.y,o.z||0)}this.setAttribute("position",new se(i,3))}else{for(let i=0,s=e.count;i<s;i++){let r=t[i];e.setXYZ(i,r.x,r.y,r.z||0)}t.length>e.count&&console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),e.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Dn);let t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new L(-1/0,-1/0,-1/0),new L(1/0,1/0,1/0));return}if(t!==void 0){if(this.boundingBox.setFromBufferAttribute(t),e)for(let i=0,s=e.length;i<s;i++){let r=e[i];Ge.setFromBufferAttribute(r),this.morphTargetsRelative?(Ee.addVectors(this.boundingBox.min,Ge.min),this.boundingBox.expandByPoint(Ee),Ee.addVectors(this.boundingBox.max,Ge.max),this.boundingBox.expandByPoint(Ee)):(this.boundingBox.expandByPoint(Ge.min),this.boundingBox.expandByPoint(Ge.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Fs);let t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new L,1/0);return}if(t){let i=this.boundingSphere.center;if(Ge.setFromBufferAttribute(t),e)for(let r=0,o=e.length;r<o;r++){let a=e[r];bs.setFromBufferAttribute(a),this.morphTargetsRelative?(Ee.addVectors(Ge.min,bs.min),Ge.expandByPoint(Ee),Ee.addVectors(Ge.max,bs.max),Ge.expandByPoint(Ee)):(Ge.expandByPoint(bs.min),Ge.expandByPoint(bs.max))}Ge.getCenter(i);let s=0;for(let r=0,o=t.count;r<o;r++)Ee.fromBufferAttribute(t,r),s=Math.max(s,i.distanceToSquared(Ee));if(e)for(let r=0,o=e.length;r<o;r++){let a=e[r],c=this.morphTargetsRelative;for(let l=0,h=a.count;l<h;l++)Ee.fromBufferAttribute(a,l),c&&(ki.fromBufferAttribute(t,l),Ee.add(ki)),s=Math.max(s,i.distanceToSquared(Ee))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let t=this.index,e=this.attributes;if(t===null||e.position===void 0||e.normal===void 0||e.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=e.position,s=e.normal,r=e.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new we(new Float32Array(4*i.count),4));let o=this.getAttribute("tangent"),a=[],c=[];for(let C=0;C<i.count;C++)a[C]=new L,c[C]=new L;let l=new L,h=new L,u=new L,d=new j,f=new j,g=new j,_=new L,m=new L;function p(C,b,y){l.fromBufferAttribute(i,C),h.fromBufferAttribute(i,b),u.fromBufferAttribute(i,y),d.fromBufferAttribute(r,C),f.fromBufferAttribute(r,b),g.fromBufferAttribute(r,y),h.sub(l),u.sub(l),f.sub(d),g.sub(d);let I=1/(f.x*g.y-g.x*f.y);isFinite(I)&&(_.copy(h).multiplyScalar(g.y).addScaledVector(u,-f.y).multiplyScalar(I),m.copy(u).multiplyScalar(f.x).addScaledVector(h,-g.x).multiplyScalar(I),a[C].add(_),a[b].add(_),a[y].add(_),c[C].add(m),c[b].add(m),c[y].add(m))}let S=this.groups;S.length===0&&(S=[{start:0,count:t.count}]);for(let C=0,b=S.length;C<b;++C){let y=S[C],I=y.start,B=y.count;for(let N=I,H=I+B;N<H;N+=3)p(t.getX(N+0),t.getX(N+1),t.getX(N+2))}let M=new L,x=new L,D=new L,T=new L;function R(C){D.fromBufferAttribute(s,C),T.copy(D);let b=a[C];M.copy(b),M.sub(D.multiplyScalar(D.dot(b))).normalize(),x.crossVectors(T,b);let I=x.dot(c[C])<0?-1:1;o.setXYZW(C,M.x,M.y,M.z,I)}for(let C=0,b=S.length;C<b;++C){let y=S[C],I=y.start,B=y.count;for(let N=I,H=I+B;N<H;N+=3)R(t.getX(N+0)),R(t.getX(N+1)),R(t.getX(N+2))}}computeVertexNormals(){let t=this.index,e=this.getAttribute("position");if(e!==void 0){let i=this.getAttribute("normal");if(i===void 0)i=new we(new Float32Array(e.count*3),3),this.setAttribute("normal",i);else for(let d=0,f=i.count;d<f;d++)i.setXYZ(d,0,0,0);let s=new L,r=new L,o=new L,a=new L,c=new L,l=new L,h=new L,u=new L;if(t)for(let d=0,f=t.count;d<f;d+=3){let g=t.getX(d+0),_=t.getX(d+1),m=t.getX(d+2);s.fromBufferAttribute(e,g),r.fromBufferAttribute(e,_),o.fromBufferAttribute(e,m),h.subVectors(o,r),u.subVectors(s,r),h.cross(u),a.fromBufferAttribute(i,g),c.fromBufferAttribute(i,_),l.fromBufferAttribute(i,m),a.add(h),c.add(h),l.add(h),i.setXYZ(g,a.x,a.y,a.z),i.setXYZ(_,c.x,c.y,c.z),i.setXYZ(m,l.x,l.y,l.z)}else for(let d=0,f=e.count;d<f;d+=3)s.fromBufferAttribute(e,d+0),r.fromBufferAttribute(e,d+1),o.fromBufferAttribute(e,d+2),h.subVectors(o,r),u.subVectors(s,r),h.cross(u),i.setXYZ(d+0,h.x,h.y,h.z),i.setXYZ(d+1,h.x,h.y,h.z),i.setXYZ(d+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let t=this.attributes.normal;for(let e=0,i=t.count;e<i;e++)Ee.fromBufferAttribute(t,e),Ee.normalize(),t.setXYZ(e,Ee.x,Ee.y,Ee.z)}toNonIndexed(){function t(a,c){let l=a.array,h=a.itemSize,u=a.normalized,d=new l.constructor(c.length*h),f=0,g=0;for(let _=0,m=c.length;_<m;_++){a.isInterleavedBufferAttribute?f=c[_]*a.data.stride+a.offset:f=c[_]*h;for(let p=0;p<h;p++)d[g++]=l[f++]}return new we(d,h,u)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let e=new n,i=this.index.array,s=this.attributes;for(let a in s){let c=s[a],l=t(c,i);e.setAttribute(a,l)}let r=this.morphAttributes;for(let a in r){let c=[],l=r[a];for(let h=0,u=l.length;h<u;h++){let d=l[h],f=t(d,i);c.push(f)}e.morphAttributes[a]=c}e.morphTargetsRelative=this.morphTargetsRelative;let o=this.groups;for(let a=0,c=o.length;a<c;a++){let l=o[a];e.addGroup(l.start,l.count,l.materialIndex)}return e}toJSON(){let t={metadata:{version:4.6,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.type,this.name!==""&&(t.name=this.name),Object.keys(this.userData).length>0&&(t.userData=this.userData),this.parameters!==void 0){let c=this.parameters;for(let l in c)c[l]!==void 0&&(t[l]=c[l]);return t}t.data={attributes:{}};let e=this.index;e!==null&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});let i=this.attributes;for(let c in i){let l=i[c];t.data.attributes[c]=l.toJSON(t.data)}let s={},r=!1;for(let c in this.morphAttributes){let l=this.morphAttributes[c],h=[];for(let u=0,d=l.length;u<d;u++){let f=l[u];h.push(f.toJSON(t.data))}h.length>0&&(s[c]=h,r=!0)}r&&(t.data.morphAttributes=s,t.data.morphTargetsRelative=this.morphTargetsRelative);let o=this.groups;o.length>0&&(t.data.groups=JSON.parse(JSON.stringify(o)));let a=this.boundingSphere;return a!==null&&(t.data.boundingSphere={center:a.center.toArray(),radius:a.radius}),t}clone(){return new this.constructor().copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let e={};this.name=t.name;let i=t.index;i!==null&&this.setIndex(i.clone(e));let s=t.attributes;for(let l in s){let h=s[l];this.setAttribute(l,h.clone(e))}let r=t.morphAttributes;for(let l in r){let h=[],u=r[l];for(let d=0,f=u.length;d<f;d++)h.push(u[d].clone(e));this.morphAttributes[l]=h}this.morphTargetsRelative=t.morphTargetsRelative;let o=t.groups;for(let l=0,h=o.length;l<h;l++){let u=o[l];this.addGroup(u.start,u.count,u.materialIndex)}let a=t.boundingBox;a!==null&&(this.boundingBox=a.clone());let c=t.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}},Dl=new de,ui=new Lr,ar=new Fs,Ul=new L,cr=new L,lr=new L,hr=new L,zo=new L,ur=new L,Nl=new L,dr=new L,Pt=class extends Ue{constructor(t=new Te,e=new Dr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),t.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}updateMorphTargets(){let e=this.geometry.morphAttributes,i=Object.keys(e);if(i.length>0){let s=e[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){let a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(t,e){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,o=i.morphTargetsRelative;e.fromBufferAttribute(s,t);let a=this.morphTargetInfluences;if(r&&a){ur.set(0,0,0);for(let c=0,l=r.length;c<l;c++){let h=a[c],u=r[c];h!==0&&(zo.fromBufferAttribute(u,t),o?ur.addScaledVector(zo,h):ur.addScaledVector(zo.sub(e),h))}e.add(ur)}return e}raycast(t,e){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),ar.copy(i.boundingSphere),ar.applyMatrix4(r),ui.copy(t.ray).recast(t.near),!(ar.containsPoint(ui.origin)===!1&&(ui.intersectSphere(ar,Ul)===null||ui.origin.distanceToSquared(Ul)>(t.far-t.near)**2))&&(Dl.copy(r).invert(),ui.copy(t.ray).applyMatrix4(Dl),!(i.boundingBox!==null&&ui.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(t,e,ui)))}_computeIntersections(t,e,i){let s,r=this.geometry,o=this.material,a=r.index,c=r.attributes.position,l=r.attributes.uv,h=r.attributes.uv1,u=r.attributes.normal,d=r.groups,f=r.drawRange;if(a!==null)if(Array.isArray(o))for(let g=0,_=d.length;g<_;g++){let m=d[g],p=o[m.materialIndex],S=Math.max(m.start,f.start),M=Math.min(a.count,Math.min(m.start+m.count,f.start+f.count));for(let x=S,D=M;x<D;x+=3){let T=a.getX(x),R=a.getX(x+1),C=a.getX(x+2);s=fr(this,p,t,i,l,h,u,T,R,C),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{let g=Math.max(0,f.start),_=Math.min(a.count,f.start+f.count);for(let m=g,p=_;m<p;m+=3){let S=a.getX(m),M=a.getX(m+1),x=a.getX(m+2);s=fr(this,o,t,i,l,h,u,S,M,x),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}else if(c!==void 0)if(Array.isArray(o))for(let g=0,_=d.length;g<_;g++){let m=d[g],p=o[m.materialIndex],S=Math.max(m.start,f.start),M=Math.min(c.count,Math.min(m.start+m.count,f.start+f.count));for(let x=S,D=M;x<D;x+=3){let T=x,R=x+1,C=x+2;s=fr(this,p,t,i,l,h,u,T,R,C),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{let g=Math.max(0,f.start),_=Math.min(c.count,f.start+f.count);for(let m=g,p=_;m<p;m+=3){let S=m,M=m+1,x=m+2;s=fr(this,o,t,i,l,h,u,S,M,x),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}}};function Yd(n,t,e,i,s,r,o,a){let c;if(t.side===Be?c=i.intersectTriangle(o,r,s,!0,a):c=i.intersectTriangle(s,r,o,t.side===$n,a),c===null)return null;dr.copy(a),dr.applyMatrix4(n.matrixWorld);let l=e.ray.origin.distanceTo(dr);return l<e.near||l>e.far?null:{distance:l,point:dr.clone(),object:n}}function fr(n,t,e,i,s,r,o,a,c,l){n.getVertexPosition(a,cr),n.getVertexPosition(c,lr),n.getVertexPosition(l,hr);let h=Yd(n,t,e,i,cr,lr,hr,Nl);if(h){let u=new L;_i.getBarycoord(Nl,cr,lr,hr,u),s&&(h.uv=_i.getInterpolatedAttribute(s,a,c,l,u,new j)),r&&(h.uv1=_i.getInterpolatedAttribute(r,a,c,l,u,new j)),o&&(h.normal=_i.getInterpolatedAttribute(o,a,c,l,u,new L),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let d={a,b:c,c:l,normal:new L,materialIndex:0};_i.getNormal(cr,lr,hr,d.normal),h.face=d,h.barycoord=u}return h}var ge=class n extends Te{constructor(t=1,e=1,i=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:i,widthSegments:s,heightSegments:r,depthSegments:o};let a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);let c=[],l=[],h=[],u=[],d=0,f=0;g("z","y","x",-1,-1,i,e,t,o,r,0),g("z","y","x",1,-1,i,e,-t,o,r,1),g("x","z","y",1,1,t,i,e,s,o,2),g("x","z","y",1,-1,t,i,-e,s,o,3),g("x","y","z",1,-1,t,e,i,s,r,4),g("x","y","z",-1,-1,t,e,-i,s,r,5),this.setIndex(c),this.setAttribute("position",new se(l,3)),this.setAttribute("normal",new se(h,3)),this.setAttribute("uv",new se(u,2));function g(_,m,p,S,M,x,D,T,R,C,b){let y=x/R,I=D/C,B=x/2,N=D/2,H=T/2,Y=R+1,V=C+1,rt=0,X=0,it=new L;for(let dt=0;dt<V;dt++){let vt=dt*I-N;for(let zt=0;zt<Y;zt++){let Kt=zt*y-B;it[_]=Kt*S,it[m]=vt*M,it[p]=H,l.push(it.x,it.y,it.z),it[_]=0,it[m]=0,it[p]=T>0?1:-1,h.push(it.x,it.y,it.z),u.push(zt/R),u.push(1-dt/C),rt+=1}}for(let dt=0;dt<C;dt++)for(let vt=0;vt<R;vt++){let zt=d+vt+Y*dt,Kt=d+vt+Y*(dt+1),P=d+(vt+1)+Y*(dt+1),k=d+(vt+1)+Y*dt;c.push(zt,Kt,k),c.push(Kt,P,k),X+=6}a.addGroup(f,X,b),f+=X,d+=rt}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}};function is(n){let t={};for(let e in n){t[e]={};for(let i in n[e]){let s=n[e][i];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),t[e][i]=null):t[e][i]=s.clone():Array.isArray(s)?t[e][i]=s.slice():t[e][i]=s}}return t}function De(n){let t={};for(let e=0;e<n.length;e++){let i=is(n[e]);for(let s in i)t[s]=i[s]}return t}function Zd(n){let t=[];for(let e=0;e<n.length;e++)t.push(n[e].clone());return t}function Nh(n){let t=n.getRenderTarget();return t===null?n.outputColorSpace:t.isXRRenderTarget===!0?t.texture.colorSpace:Jt.workingColorSpace}var $d={clone:is,merge:De},Jd=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Kd=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,dn=class extends Si{static get type(){return"ShaderMaterial"}constructor(t){super(),this.isShaderMaterial=!0,this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=Jd,this.fragmentShader=Kd,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=is(t.uniforms),this.uniformsGroups=Zd(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){let e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(let s in this.uniforms){let o=this.uniforms[s].value;o&&o.isTexture?e.uniforms[s]={type:"t",value:o.toJSON(t).uuid}:o&&o.isColor?e.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?e.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?e.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?e.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?e.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?e.uniforms[s]={type:"m4",value:o.toArray()}:e.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader,e.lights=this.lights,e.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(e.extensions=i),e}},Fr=class extends Ue{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new de,this.projectionMatrix=new de,this.projectionMatrixInverse=new de,this.coordinateSystem=Cn}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this.coordinateSystem=t.coordinateSystem,this}getWorldDirection(t){return super.getWorldDirection(t).negate()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}},Wn=new L,Fl=new j,Ol=new j,Pe=class extends Fr{constructor(t=50,e=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){let e=.5*this.getFilmHeight()/t;this.fov=Ns*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){let t=Math.tan(Rs*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return Ns*2*Math.atan(Math.tan(Rs*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(t,e,i){Wn.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),e.set(Wn.x,Wn.y).multiplyScalar(-t/Wn.z),Wn.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(Wn.x,Wn.y).multiplyScalar(-t/Wn.z)}getViewSize(t,e){return this.getViewBounds(t,Fl,Ol),e.subVectors(Ol,Fl)}setViewOffset(t,e,i,s,r,o){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let t=this.near,e=t*Math.tan(Rs*.5*this.fov)/this.zoom,i=2*e,s=this.aspect*i,r=-.5*s,o=this.view;if(this.view!==null&&this.view.enabled){let c=o.fullWidth,l=o.fullHeight;r+=o.offsetX*s/c,e-=o.offsetY*i/l,s*=o.width/c,i*=o.height/l}let a=this.filmOffset;a!==0&&(r+=t*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,e,e-i,t,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){let e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}},Vi=-90,Gi=1,Ha=class extends Ue{constructor(t,e,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Pe(Vi,Gi,t,e);s.layers=this.layers,this.add(s);let r=new Pe(Vi,Gi,t,e);r.layers=this.layers,this.add(r);let o=new Pe(Vi,Gi,t,e);o.layers=this.layers,this.add(o);let a=new Pe(Vi,Gi,t,e);a.layers=this.layers,this.add(a);let c=new Pe(Vi,Gi,t,e);c.layers=this.layers,this.add(c);let l=new Pe(Vi,Gi,t,e);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){let t=this.coordinateSystem,e=this.children.concat(),[i,s,r,o,a,c]=e;for(let l of e)this.remove(l);if(t===Cn)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(t===Rr)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+t);for(let l of e)this.add(l),l.updateMatrixWorld()}update(t,e){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==t.coordinateSystem&&(this.coordinateSystem=t.coordinateSystem,this.updateCoordinateSystem());let[r,o,a,c,l,h]=this.children,u=t.getRenderTarget(),d=t.getActiveCubeFace(),f=t.getActiveMipmapLevel(),g=t.xr.enabled;t.xr.enabled=!1;let _=i.texture.generateMipmaps;i.texture.generateMipmaps=!1,t.setRenderTarget(i,0,s),t.render(e,r),t.setRenderTarget(i,1,s),t.render(e,o),t.setRenderTarget(i,2,s),t.render(e,a),t.setRenderTarget(i,3,s),t.render(e,c),t.setRenderTarget(i,4,s),t.render(e,l),i.texture.generateMipmaps=_,t.setRenderTarget(i,5,s),t.render(e,h),t.setRenderTarget(u,d,f),t.xr.enabled=g,i.texture.needsPMREMUpdate=!0}},Or=class extends We{constructor(t,e,i,s,r,o,a,c,l,h){t=t!==void 0?t:[],e=e!==void 0?e:ji,super(t,e,i,s,r,o,a,c,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}},ka=class extends Ln{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;let i={width:t,height:t,depth:1},s=[i,i,i,i,i,i];this.texture=new Or(s,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=e.generateMipmaps!==void 0?e.generateMipmaps:!1,this.texture.minFilter=e.minFilter!==void 0?e.minFilter:hn}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.colorSpace=e.colorSpace,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new ge(5,5,5),r=new dn({name:"CubemapFromEquirect",uniforms:is(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:Be,blending:qn});r.uniforms.tEquirect.value=e;let o=new Pt(s,r),a=e.minFilter;return e.minFilter===vi&&(e.minFilter=hn),new Ha(1,10,this).update(t,o),e.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(t,e,i,s){let r=t.getRenderTarget();for(let o=0;o<6;o++)t.setRenderTarget(this,o),t.clear(e,i,s);t.setRenderTarget(r)}},Ho=new L,Qd=new L,jd=new Wt,An=class{constructor(t=new L(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,i,s){return this.normal.set(t,e,i),this.constant=s,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,i){let s=Ho.subVectors(i,e).cross(Qd.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(s,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){let t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e){let i=t.delta(Ho),s=this.normal.dot(i);if(s===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;let r=-(t.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:e.copy(t.start).addScaledVector(i,r)}intersectsLine(t){let e=this.distanceToPoint(t.start),i=this.distanceToPoint(t.end);return e<0&&i>0||i<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){let i=e||jd.getNormalMatrix(t),s=this.coplanarPoint(Ho).applyMatrix4(t),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}},di=new Fs,pr=new L,Bs=class{constructor(t=new An,e=new An,i=new An,s=new An,r=new An,o=new An){this.planes=[t,e,i,s,r,o]}set(t,e,i,s,r,o){let a=this.planes;return a[0].copy(t),a[1].copy(e),a[2].copy(i),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(t){let e=this.planes;for(let i=0;i<6;i++)e[i].copy(t.planes[i]);return this}setFromProjectionMatrix(t,e=Cn){let i=this.planes,s=t.elements,r=s[0],o=s[1],a=s[2],c=s[3],l=s[4],h=s[5],u=s[6],d=s[7],f=s[8],g=s[9],_=s[10],m=s[11],p=s[12],S=s[13],M=s[14],x=s[15];if(i[0].setComponents(c-r,d-l,m-f,x-p).normalize(),i[1].setComponents(c+r,d+l,m+f,x+p).normalize(),i[2].setComponents(c+o,d+h,m+g,x+S).normalize(),i[3].setComponents(c-o,d-h,m-g,x-S).normalize(),i[4].setComponents(c-a,d-u,m-_,x-M).normalize(),e===Cn)i[5].setComponents(c+a,d+u,m+_,x+M).normalize();else if(e===Rr)i[5].setComponents(a,u,_,M).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+e);return this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),di.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{let e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),di.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(di)}intersectsSprite(t){return di.center.set(0,0,0),di.radius=.7071067811865476,di.applyMatrix4(t.matrixWorld),this.intersectsSphere(di)}intersectsSphere(t){let e=this.planes,i=t.center,s=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(t){let e=this.planes;for(let i=0;i<6;i++){let s=e[i];if(pr.x=s.normal.x>0?t.max.x:t.min.x,pr.y=s.normal.y>0?t.max.y:t.min.y,pr.z=s.normal.z>0?t.max.z:t.min.z,s.distanceToPoint(pr)<0)return!1}return!0}containsPoint(t){let e=this.planes;for(let i=0;i<6;i++)if(e[i].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};function Fh(){let n=null,t=!1,e=null,i=null;function s(r,o){e(r,o),i=n.requestAnimationFrame(s)}return{start:function(){t!==!0&&e!==null&&(i=n.requestAnimationFrame(s),t=!0)},stop:function(){n.cancelAnimationFrame(i),t=!1},setAnimationLoop:function(r){e=r},setContext:function(r){n=r}}}function tf(n){let t=new WeakMap;function e(a,c){let l=a.array,h=a.usage,u=l.byteLength,d=n.createBuffer();n.bindBuffer(c,d),n.bufferData(c,l,h),a.onUploadCallback();let f;if(l instanceof Float32Array)f=n.FLOAT;else if(l instanceof Uint16Array)a.isFloat16BufferAttribute?f=n.HALF_FLOAT:f=n.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=n.SHORT;else if(l instanceof Uint32Array)f=n.UNSIGNED_INT;else if(l instanceof Int32Array)f=n.INT;else if(l instanceof Int8Array)f=n.BYTE;else if(l instanceof Uint8Array)f=n.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=n.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:d,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:a.version,size:u}}function i(a,c,l){let h=c.array,u=c.updateRanges;if(n.bindBuffer(l,a),u.length===0)n.bufferSubData(l,0,h);else{u.sort((f,g)=>f.start-g.start);let d=0;for(let f=1;f<u.length;f++){let g=u[d],_=u[f];_.start<=g.start+g.count+1?g.count=Math.max(g.count,_.start+_.count-g.start):(++d,u[d]=_)}u.length=d+1;for(let f=0,g=u.length;f<g;f++){let _=u[f];n.bufferSubData(l,_.start*h.BYTES_PER_ELEMENT,h,_.start,_.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),t.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);let c=t.get(a);c&&(n.deleteBuffer(c.buffer),t.delete(a))}function o(a,c){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){let h=t.get(a);(!h||h.version<a.version)&&t.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}let l=t.get(a);if(l===void 0)t.set(a,e(a,c));else if(l.version<a.version){if(l.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(l.buffer,a,c),l.version=a.version}}return{get:s,remove:r,update:o}}var Un=class n extends Te{constructor(t=1,e=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:i,heightSegments:s};let r=t/2,o=e/2,a=Math.floor(i),c=Math.floor(s),l=a+1,h=c+1,u=t/a,d=e/c,f=[],g=[],_=[],m=[];for(let p=0;p<h;p++){let S=p*d-o;for(let M=0;M<l;M++){let x=M*u-r;g.push(x,-S,0),_.push(0,0,1),m.push(M/a),m.push(1-p/c)}}for(let p=0;p<c;p++)for(let S=0;S<a;S++){let M=S+l*p,x=S+l*(p+1),D=S+1+l*(p+1),T=S+1+l*p;f.push(M,x,T),f.push(x,D,T)}this.setIndex(f),this.setAttribute("position",new se(g,3)),this.setAttribute("normal",new se(_,3)),this.setAttribute("uv",new se(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.width,t.height,t.widthSegments,t.heightSegments)}},ef=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,nf=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,sf=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,rf=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,of=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,af=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,cf=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,lf=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,hf=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,uf=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,df=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,ff=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,pf=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,mf=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,gf=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,_f=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,xf=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,yf=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,vf=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Mf=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Sf=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Ef=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,bf=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,wf=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Tf=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Af=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Rf=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Cf=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,If=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Pf=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Lf="gl_FragColor = linearToOutputTexel( gl_FragColor );",Df=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Uf=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,Nf=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,Ff=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,Of=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,Bf=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,zf=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,Hf=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,kf=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Vf=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,Gf=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,Wf=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Xf=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,qf=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Yf=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,Zf=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,$f=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Jf=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Kf=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Qf=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,jf=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,tp=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,ep=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,np=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,ip=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,sp=`#if defined( USE_LOGDEPTHBUF )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,rp=`#if defined( USE_LOGDEPTHBUF )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,op=`#ifdef USE_LOGDEPTHBUF
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,ap=`#ifdef USE_LOGDEPTHBUF
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,cp=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,lp=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,hp=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,up=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,dp=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,fp=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,pp=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,mp=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,gp=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,_p=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,xp=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,yp=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,vp=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Mp=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Sp=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Ep=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,bp=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,wp=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Tp=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Ap=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Rp=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,Cp=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Ip=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,Pp=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Lp=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Dp=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Up=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,Np=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,Fp=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,Op=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,Bp=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,zp=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,Hp=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,kp=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,Vp=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,Gp=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,Wp=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,Xp=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,qp=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Yp=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,Zp=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,$p=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,Jp=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
		
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
		
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		
		#else
		
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Kp=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Qp=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,jp=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,tm=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,em=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,nm=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,im=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,sm=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,rm=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,om=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,am=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,cm=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,lm=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,hm=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,um=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,dm=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,fm=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,pm=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,mm=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,gm=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,_m=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,xm=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ym=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,vm=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Mm=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Sm=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Em=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,bm=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,wm=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Tm=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Am=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Rm=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Cm=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Im=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Pm=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Lm=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Dm=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Um=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,qt={alphahash_fragment:ef,alphahash_pars_fragment:nf,alphamap_fragment:sf,alphamap_pars_fragment:rf,alphatest_fragment:of,alphatest_pars_fragment:af,aomap_fragment:cf,aomap_pars_fragment:lf,batching_pars_vertex:hf,batching_vertex:uf,begin_vertex:df,beginnormal_vertex:ff,bsdfs:pf,iridescence_fragment:mf,bumpmap_pars_fragment:gf,clipping_planes_fragment:_f,clipping_planes_pars_fragment:xf,clipping_planes_pars_vertex:yf,clipping_planes_vertex:vf,color_fragment:Mf,color_pars_fragment:Sf,color_pars_vertex:Ef,color_vertex:bf,common:wf,cube_uv_reflection_fragment:Tf,defaultnormal_vertex:Af,displacementmap_pars_vertex:Rf,displacementmap_vertex:Cf,emissivemap_fragment:If,emissivemap_pars_fragment:Pf,colorspace_fragment:Lf,colorspace_pars_fragment:Df,envmap_fragment:Uf,envmap_common_pars_fragment:Nf,envmap_pars_fragment:Ff,envmap_pars_vertex:Of,envmap_physical_pars_fragment:Zf,envmap_vertex:Bf,fog_vertex:zf,fog_pars_vertex:Hf,fog_fragment:kf,fog_pars_fragment:Vf,gradientmap_pars_fragment:Gf,lightmap_pars_fragment:Wf,lights_lambert_fragment:Xf,lights_lambert_pars_fragment:qf,lights_pars_begin:Yf,lights_toon_fragment:$f,lights_toon_pars_fragment:Jf,lights_phong_fragment:Kf,lights_phong_pars_fragment:Qf,lights_physical_fragment:jf,lights_physical_pars_fragment:tp,lights_fragment_begin:ep,lights_fragment_maps:np,lights_fragment_end:ip,logdepthbuf_fragment:sp,logdepthbuf_pars_fragment:rp,logdepthbuf_pars_vertex:op,logdepthbuf_vertex:ap,map_fragment:cp,map_pars_fragment:lp,map_particle_fragment:hp,map_particle_pars_fragment:up,metalnessmap_fragment:dp,metalnessmap_pars_fragment:fp,morphinstance_vertex:pp,morphcolor_vertex:mp,morphnormal_vertex:gp,morphtarget_pars_vertex:_p,morphtarget_vertex:xp,normal_fragment_begin:yp,normal_fragment_maps:vp,normal_pars_fragment:Mp,normal_pars_vertex:Sp,normal_vertex:Ep,normalmap_pars_fragment:bp,clearcoat_normal_fragment_begin:wp,clearcoat_normal_fragment_maps:Tp,clearcoat_pars_fragment:Ap,iridescence_pars_fragment:Rp,opaque_fragment:Cp,packing:Ip,premultiplied_alpha_fragment:Pp,project_vertex:Lp,dithering_fragment:Dp,dithering_pars_fragment:Up,roughnessmap_fragment:Np,roughnessmap_pars_fragment:Fp,shadowmap_pars_fragment:Op,shadowmap_pars_vertex:Bp,shadowmap_vertex:zp,shadowmask_pars_fragment:Hp,skinbase_vertex:kp,skinning_pars_vertex:Vp,skinning_vertex:Gp,skinnormal_vertex:Wp,specularmap_fragment:Xp,specularmap_pars_fragment:qp,tonemapping_fragment:Yp,tonemapping_pars_fragment:Zp,transmission_fragment:$p,transmission_pars_fragment:Jp,uv_pars_fragment:Kp,uv_pars_vertex:Qp,uv_vertex:jp,worldpos_vertex:tm,background_vert:em,background_frag:nm,backgroundCube_vert:im,backgroundCube_frag:sm,cube_vert:rm,cube_frag:om,depth_vert:am,depth_frag:cm,distanceRGBA_vert:lm,distanceRGBA_frag:hm,equirect_vert:um,equirect_frag:dm,linedashed_vert:fm,linedashed_frag:pm,meshbasic_vert:mm,meshbasic_frag:gm,meshlambert_vert:_m,meshlambert_frag:xm,meshmatcap_vert:ym,meshmatcap_frag:vm,meshnormal_vert:Mm,meshnormal_frag:Sm,meshphong_vert:Em,meshphong_frag:bm,meshphysical_vert:wm,meshphysical_frag:Tm,meshtoon_vert:Am,meshtoon_frag:Rm,points_vert:Cm,points_frag:Im,shadow_vert:Pm,shadow_frag:Lm,sprite_vert:Dm,sprite_frag:Um},ut={common:{diffuse:{value:new kt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Wt},alphaMap:{value:null},alphaMapTransform:{value:new Wt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Wt}},envmap:{envMap:{value:null},envMapRotation:{value:new Wt},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Wt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Wt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Wt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Wt},normalScale:{value:new j(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Wt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Wt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Wt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Wt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new kt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new kt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Wt},alphaTest:{value:0},uvTransform:{value:new Wt}},sprite:{diffuse:{value:new kt(16777215)},opacity:{value:1},center:{value:new j(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Wt},alphaMap:{value:null},alphaMapTransform:{value:new Wt},alphaTest:{value:0}}},ln={basic:{uniforms:De([ut.common,ut.specularmap,ut.envmap,ut.aomap,ut.lightmap,ut.fog]),vertexShader:qt.meshbasic_vert,fragmentShader:qt.meshbasic_frag},lambert:{uniforms:De([ut.common,ut.specularmap,ut.envmap,ut.aomap,ut.lightmap,ut.emissivemap,ut.bumpmap,ut.normalmap,ut.displacementmap,ut.fog,ut.lights,{emissive:{value:new kt(0)}}]),vertexShader:qt.meshlambert_vert,fragmentShader:qt.meshlambert_frag},phong:{uniforms:De([ut.common,ut.specularmap,ut.envmap,ut.aomap,ut.lightmap,ut.emissivemap,ut.bumpmap,ut.normalmap,ut.displacementmap,ut.fog,ut.lights,{emissive:{value:new kt(0)},specular:{value:new kt(1118481)},shininess:{value:30}}]),vertexShader:qt.meshphong_vert,fragmentShader:qt.meshphong_frag},standard:{uniforms:De([ut.common,ut.envmap,ut.aomap,ut.lightmap,ut.emissivemap,ut.bumpmap,ut.normalmap,ut.displacementmap,ut.roughnessmap,ut.metalnessmap,ut.fog,ut.lights,{emissive:{value:new kt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:qt.meshphysical_vert,fragmentShader:qt.meshphysical_frag},toon:{uniforms:De([ut.common,ut.aomap,ut.lightmap,ut.emissivemap,ut.bumpmap,ut.normalmap,ut.displacementmap,ut.gradientmap,ut.fog,ut.lights,{emissive:{value:new kt(0)}}]),vertexShader:qt.meshtoon_vert,fragmentShader:qt.meshtoon_frag},matcap:{uniforms:De([ut.common,ut.bumpmap,ut.normalmap,ut.displacementmap,ut.fog,{matcap:{value:null}}]),vertexShader:qt.meshmatcap_vert,fragmentShader:qt.meshmatcap_frag},points:{uniforms:De([ut.points,ut.fog]),vertexShader:qt.points_vert,fragmentShader:qt.points_frag},dashed:{uniforms:De([ut.common,ut.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:qt.linedashed_vert,fragmentShader:qt.linedashed_frag},depth:{uniforms:De([ut.common,ut.displacementmap]),vertexShader:qt.depth_vert,fragmentShader:qt.depth_frag},normal:{uniforms:De([ut.common,ut.bumpmap,ut.normalmap,ut.displacementmap,{opacity:{value:1}}]),vertexShader:qt.meshnormal_vert,fragmentShader:qt.meshnormal_frag},sprite:{uniforms:De([ut.sprite,ut.fog]),vertexShader:qt.sprite_vert,fragmentShader:qt.sprite_frag},background:{uniforms:{uvTransform:{value:new Wt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:qt.background_vert,fragmentShader:qt.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Wt}},vertexShader:qt.backgroundCube_vert,fragmentShader:qt.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:qt.cube_vert,fragmentShader:qt.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:qt.equirect_vert,fragmentShader:qt.equirect_frag},distanceRGBA:{uniforms:De([ut.common,ut.displacementmap,{referencePosition:{value:new L},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:qt.distanceRGBA_vert,fragmentShader:qt.distanceRGBA_frag},shadow:{uniforms:De([ut.lights,ut.fog,{color:{value:new kt(0)},opacity:{value:1}}]),vertexShader:qt.shadow_vert,fragmentShader:qt.shadow_frag}};ln.physical={uniforms:De([ln.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Wt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Wt},clearcoatNormalScale:{value:new j(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Wt},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Wt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Wt},sheen:{value:0},sheenColor:{value:new kt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Wt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Wt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Wt},transmissionSamplerSize:{value:new j},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Wt},attenuationDistance:{value:0},attenuationColor:{value:new kt(0)},specularColor:{value:new kt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Wt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Wt},anisotropyVector:{value:new j},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Wt}}]),vertexShader:qt.meshphysical_vert,fragmentShader:qt.meshphysical_frag};var mr={r:0,b:0,g:0},fi=new un,Nm=new de;function Fm(n,t,e,i,s,r,o){let a=new kt(0),c=r===!0?0:1,l,h,u=null,d=0,f=null;function g(S){let M=S.isScene===!0?S.background:null;return M&&M.isTexture&&(M=(S.backgroundBlurriness>0?e:t).get(M)),M}function _(S){let M=!1,x=g(S);x===null?p(a,c):x&&x.isColor&&(p(x,1),M=!0);let D=n.xr.getEnvironmentBlendMode();D==="additive"?i.buffers.color.setClear(0,0,0,1,o):D==="alpha-blend"&&i.buffers.color.setClear(0,0,0,0,o),(n.autoClear||M)&&(i.buffers.depth.setTest(!0),i.buffers.depth.setMask(!0),i.buffers.color.setMask(!0),n.clear(n.autoClearColor,n.autoClearDepth,n.autoClearStencil))}function m(S,M){let x=g(M);x&&(x.isCubeTexture||x.mapping===io)?(h===void 0&&(h=new Pt(new ge(1,1,1),new dn({name:"BackgroundCubeMaterial",uniforms:is(ln.backgroundCube.uniforms),vertexShader:ln.backgroundCube.vertexShader,fragmentShader:ln.backgroundCube.fragmentShader,side:Be,depthTest:!1,depthWrite:!1,fog:!1})),h.geometry.deleteAttribute("normal"),h.geometry.deleteAttribute("uv"),h.onBeforeRender=function(D,T,R){this.matrixWorld.copyPosition(R.matrixWorld)},Object.defineProperty(h.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(h)),fi.copy(M.backgroundRotation),fi.x*=-1,fi.y*=-1,fi.z*=-1,x.isCubeTexture&&x.isRenderTargetTexture===!1&&(fi.y*=-1,fi.z*=-1),h.material.uniforms.envMap.value=x,h.material.uniforms.flipEnvMap.value=x.isCubeTexture&&x.isRenderTargetTexture===!1?-1:1,h.material.uniforms.backgroundBlurriness.value=M.backgroundBlurriness,h.material.uniforms.backgroundIntensity.value=M.backgroundIntensity,h.material.uniforms.backgroundRotation.value.setFromMatrix4(Nm.makeRotationFromEuler(fi)),h.material.toneMapped=Jt.getTransfer(x.colorSpace)!==ne,(u!==x||d!==x.version||f!==n.toneMapping)&&(h.material.needsUpdate=!0,u=x,d=x.version,f=n.toneMapping),h.layers.enableAll(),S.unshift(h,h.geometry,h.material,0,0,null)):x&&x.isTexture&&(l===void 0&&(l=new Pt(new Un(2,2),new dn({name:"BackgroundMaterial",uniforms:is(ln.background.uniforms),vertexShader:ln.background.vertexShader,fragmentShader:ln.background.fragmentShader,side:$n,depthTest:!1,depthWrite:!1,fog:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(l)),l.material.uniforms.t2D.value=x,l.material.uniforms.backgroundIntensity.value=M.backgroundIntensity,l.material.toneMapped=Jt.getTransfer(x.colorSpace)!==ne,x.matrixAutoUpdate===!0&&x.updateMatrix(),l.material.uniforms.uvTransform.value.copy(x.matrix),(u!==x||d!==x.version||f!==n.toneMapping)&&(l.material.needsUpdate=!0,u=x,d=x.version,f=n.toneMapping),l.layers.enableAll(),S.unshift(l,l.geometry,l.material,0,0,null))}function p(S,M){S.getRGB(mr,Nh(n)),i.buffers.color.setClear(mr.r,mr.g,mr.b,M,o)}return{getClearColor:function(){return a},setClearColor:function(S,M=1){a.set(S),c=M,p(a,c)},getClearAlpha:function(){return c},setClearAlpha:function(S){c=S,p(a,c)},render:_,addToRenderList:m}}function Om(n,t){let e=n.getParameter(n.MAX_VERTEX_ATTRIBS),i={},s=d(null),r=s,o=!1;function a(y,I,B,N,H){let Y=!1,V=u(N,B,I);r!==V&&(r=V,l(r.object)),Y=f(y,N,B,H),Y&&g(y,N,B,H),H!==null&&t.update(H,n.ELEMENT_ARRAY_BUFFER),(Y||o)&&(o=!1,x(y,I,B,N),H!==null&&n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,t.get(H).buffer))}function c(){return n.createVertexArray()}function l(y){return n.bindVertexArray(y)}function h(y){return n.deleteVertexArray(y)}function u(y,I,B){let N=B.wireframe===!0,H=i[y.id];H===void 0&&(H={},i[y.id]=H);let Y=H[I.id];Y===void 0&&(Y={},H[I.id]=Y);let V=Y[N];return V===void 0&&(V=d(c()),Y[N]=V),V}function d(y){let I=[],B=[],N=[];for(let H=0;H<e;H++)I[H]=0,B[H]=0,N[H]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:I,enabledAttributes:B,attributeDivisors:N,object:y,attributes:{},index:null}}function f(y,I,B,N){let H=r.attributes,Y=I.attributes,V=0,rt=B.getAttributes();for(let X in rt)if(rt[X].location>=0){let dt=H[X],vt=Y[X];if(vt===void 0&&(X==="instanceMatrix"&&y.instanceMatrix&&(vt=y.instanceMatrix),X==="instanceColor"&&y.instanceColor&&(vt=y.instanceColor)),dt===void 0||dt.attribute!==vt||vt&&dt.data!==vt.data)return!0;V++}return r.attributesNum!==V||r.index!==N}function g(y,I,B,N){let H={},Y=I.attributes,V=0,rt=B.getAttributes();for(let X in rt)if(rt[X].location>=0){let dt=Y[X];dt===void 0&&(X==="instanceMatrix"&&y.instanceMatrix&&(dt=y.instanceMatrix),X==="instanceColor"&&y.instanceColor&&(dt=y.instanceColor));let vt={};vt.attribute=dt,dt&&dt.data&&(vt.data=dt.data),H[X]=vt,V++}r.attributes=H,r.attributesNum=V,r.index=N}function _(){let y=r.newAttributes;for(let I=0,B=y.length;I<B;I++)y[I]=0}function m(y){p(y,0)}function p(y,I){let B=r.newAttributes,N=r.enabledAttributes,H=r.attributeDivisors;B[y]=1,N[y]===0&&(n.enableVertexAttribArray(y),N[y]=1),H[y]!==I&&(n.vertexAttribDivisor(y,I),H[y]=I)}function S(){let y=r.newAttributes,I=r.enabledAttributes;for(let B=0,N=I.length;B<N;B++)I[B]!==y[B]&&(n.disableVertexAttribArray(B),I[B]=0)}function M(y,I,B,N,H,Y,V){V===!0?n.vertexAttribIPointer(y,I,B,H,Y):n.vertexAttribPointer(y,I,B,N,H,Y)}function x(y,I,B,N){_();let H=N.attributes,Y=B.getAttributes(),V=I.defaultAttributeValues;for(let rt in Y){let X=Y[rt];if(X.location>=0){let it=H[rt];if(it===void 0&&(rt==="instanceMatrix"&&y.instanceMatrix&&(it=y.instanceMatrix),rt==="instanceColor"&&y.instanceColor&&(it=y.instanceColor)),it!==void 0){let dt=it.normalized,vt=it.itemSize,zt=t.get(it);if(zt===void 0)continue;let Kt=zt.buffer,P=zt.type,k=zt.bytesPerElement,nt=P===n.INT||P===n.UNSIGNED_INT||it.gpuType===Tc;if(it.isInterleavedBufferAttribute){let tt=it.data,wt=tt.stride,Dt=it.offset;if(tt.isInstancedInterleavedBuffer){for(let Lt=0;Lt<X.locationSize;Lt++)p(X.location+Lt,tt.meshPerAttribute);y.isInstancedMesh!==!0&&N._maxInstanceCount===void 0&&(N._maxInstanceCount=tt.meshPerAttribute*tt.count)}else for(let Lt=0;Lt<X.locationSize;Lt++)m(X.location+Lt);n.bindBuffer(n.ARRAY_BUFFER,Kt);for(let Lt=0;Lt<X.locationSize;Lt++)M(X.location+Lt,vt/X.locationSize,P,dt,wt*k,(Dt+vt/X.locationSize*Lt)*k,nt)}else{if(it.isInstancedBufferAttribute){for(let tt=0;tt<X.locationSize;tt++)p(X.location+tt,it.meshPerAttribute);y.isInstancedMesh!==!0&&N._maxInstanceCount===void 0&&(N._maxInstanceCount=it.meshPerAttribute*it.count)}else for(let tt=0;tt<X.locationSize;tt++)m(X.location+tt);n.bindBuffer(n.ARRAY_BUFFER,Kt);for(let tt=0;tt<X.locationSize;tt++)M(X.location+tt,vt/X.locationSize,P,dt,vt*k,vt/X.locationSize*tt*k,nt)}}else if(V!==void 0){let dt=V[rt];if(dt!==void 0)switch(dt.length){case 2:n.vertexAttrib2fv(X.location,dt);break;case 3:n.vertexAttrib3fv(X.location,dt);break;case 4:n.vertexAttrib4fv(X.location,dt);break;default:n.vertexAttrib1fv(X.location,dt)}}}}S()}function D(){C();for(let y in i){let I=i[y];for(let B in I){let N=I[B];for(let H in N)h(N[H].object),delete N[H];delete I[B]}delete i[y]}}function T(y){if(i[y.id]===void 0)return;let I=i[y.id];for(let B in I){let N=I[B];for(let H in N)h(N[H].object),delete N[H];delete I[B]}delete i[y.id]}function R(y){for(let I in i){let B=i[I];if(B[y.id]===void 0)continue;let N=B[y.id];for(let H in N)h(N[H].object),delete N[H];delete B[y.id]}}function C(){b(),o=!0,r!==s&&(r=s,l(r.object))}function b(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:C,resetDefaultState:b,dispose:D,releaseStatesOfGeometry:T,releaseStatesOfProgram:R,initAttributes:_,enableAttribute:m,disableUnusedAttributes:S}}function Bm(n,t,e){let i;function s(l){i=l}function r(l,h){n.drawArrays(i,l,h),e.update(h,i,1)}function o(l,h,u){u!==0&&(n.drawArraysInstanced(i,l,h,u),e.update(h,i,u))}function a(l,h,u){if(u===0)return;t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,l,0,h,0,u);let f=0;for(let g=0;g<u;g++)f+=h[g];e.update(f,i,1)}function c(l,h,u,d){if(u===0)return;let f=t.get("WEBGL_multi_draw");if(f===null)for(let g=0;g<l.length;g++)o(l[g],h[g],d[g]);else{f.multiDrawArraysInstancedWEBGL(i,l,0,h,0,d,0,u);let g=0;for(let _=0;_<u;_++)g+=h[_]*d[_];e.update(g,i,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=c}function zm(n,t,e,i){let s;function r(){if(s!==void 0)return s;if(t.has("EXT_texture_filter_anisotropic")===!0){let R=t.get("EXT_texture_filter_anisotropic");s=n.getParameter(R.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(R){return!(R!==sn&&i.convert(R)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(R){let C=R===Gs&&(t.has("EXT_color_buffer_half_float")||t.has("EXT_color_buffer_float"));return!(R!==Pn&&i.convert(R)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE)&&R!==Rn&&!C)}function c(R){if(R==="highp"){if(n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.HIGH_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.HIGH_FLOAT).precision>0)return"highp";R="mediump"}return R==="mediump"&&n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.MEDIUM_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=e.precision!==void 0?e.precision:"highp",h=c(l);h!==l&&(console.warn("THREE.WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);let u=e.logarithmicDepthBuffer===!0,d=e.reverseDepthBuffer===!0&&t.has("EXT_clip_control"),f=n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS),g=n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS),_=n.getParameter(n.MAX_TEXTURE_SIZE),m=n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE),p=n.getParameter(n.MAX_VERTEX_ATTRIBS),S=n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS),M=n.getParameter(n.MAX_VARYING_VECTORS),x=n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS),D=g>0,T=n.getParameter(n.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:o,textureTypeReadable:a,precision:l,logarithmicDepthBuffer:u,reverseDepthBuffer:d,maxTextures:f,maxVertexTextures:g,maxTextureSize:_,maxCubemapSize:m,maxAttributes:p,maxVertexUniforms:S,maxVaryings:M,maxFragmentUniforms:x,vertexTextures:D,maxSamples:T}}function Hm(n){let t=this,e=null,i=0,s=!1,r=!1,o=new An,a=new Wt,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(u,d){let f=u.length!==0||d||i!==0||s;return s=d,i=u.length,f},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(u,d){e=h(u,d,0)},this.setState=function(u,d,f){let g=u.clippingPlanes,_=u.clipIntersection,m=u.clipShadows,p=n.get(u);if(!s||g===null||g.length===0||r&&!m)r?h(null):l();else{let S=r?0:i,M=S*4,x=p.clippingState||null;c.value=x,x=h(g,d,M,f);for(let D=0;D!==M;++D)x[D]=e[D];p.clippingState=x,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=S}};function l(){c.value!==e&&(c.value=e,c.needsUpdate=i>0),t.numPlanes=i,t.numIntersection=0}function h(u,d,f,g){let _=u!==null?u.length:0,m=null;if(_!==0){if(m=c.value,g!==!0||m===null){let p=f+_*4,S=d.matrixWorldInverse;a.getNormalMatrix(S),(m===null||m.length<p)&&(m=new Float32Array(p));for(let M=0,x=f;M!==_;++M,x+=4)o.copy(u[M]).applyMatrix4(S,a),o.normal.toArray(m,x),m[x+3]=o.constant}c.value=m,c.needsUpdate=!0}return t.numPlanes=_,t.numIntersection=0,m}}function km(n){let t=new WeakMap;function e(o,a){return a===aa?o.mapping=ji:a===ca&&(o.mapping=ts),o}function i(o){if(o&&o.isTexture){let a=o.mapping;if(a===aa||a===ca)if(t.has(o)){let c=t.get(o).texture;return e(c,o.mapping)}else{let c=o.image;if(c&&c.height>0){let l=new ka(c.height);return l.fromEquirectangularTexture(n,o),t.set(o,l),o.addEventListener("dispose",s),e(l.texture,o.mapping)}else return null}}return o}function s(o){let a=o.target;a.removeEventListener("dispose",s);let c=t.get(a);c!==void 0&&(t.delete(a),c.dispose())}function r(){t=new WeakMap}return{get:i,dispose:r}}var Br=class extends Fr{constructor(t=-1,e=1,i=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=i,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,i,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-t,o=i+t,a=s+e,c=s-e;if(this.view!==null&&this.view.enabled){let l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,o=r+l*this.view.width,a-=h*this.view.offsetY,c=a-h*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,c,this.near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){let e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}},qi=4,Bl=[.125,.215,.35,.446,.526,.582],xi=20,ko=new Br,zl=new kt,Vo=null,Go=0,Wo=0,Xo=!1,mi=(1+Math.sqrt(5))/2,Wi=1/mi,Hl=[new L(-mi,Wi,0),new L(mi,Wi,0),new L(-Wi,0,mi),new L(Wi,0,mi),new L(0,mi,-Wi),new L(0,mi,Wi),new L(-1,1,-1),new L(1,1,-1),new L(-1,1,1),new L(1,1,1)],zr=class{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,i=.1,s=100){Vo=this._renderer.getRenderTarget(),Go=this._renderer.getActiveCubeFace(),Wo=this._renderer.getActiveMipmapLevel(),Xo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(256);let r=this._allocateTargets();return r.depthBuffer=!0,this._sceneToCubeUV(t,i,s,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Gl(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Vl(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(t){this._lodMax=Math.floor(Math.log2(t)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let t=0;t<this._lodPlanes.length;t++)this._lodPlanes[t].dispose()}_cleanup(t){this._renderer.setRenderTarget(Vo,Go,Wo),this._renderer.xr.enabled=Xo,t.scissorTest=!1,gr(t,0,0,t.width,t.height)}_fromTexture(t,e){t.mapping===ji||t.mapping===ts?this._setSize(t.image.length===0?16:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4),Vo=this._renderer.getRenderTarget(),Go=this._renderer.getActiveCubeFace(),Wo=this._renderer.getActiveMipmapLevel(),Xo=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let i=e||this._allocateTargets();return this._textureToCubeUV(t,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){let t=3*Math.max(this._cubeSize,112),e=4*this._cubeSize,i={magFilter:hn,minFilter:hn,generateMipmaps:!1,type:Gs,format:sn,colorSpace:cs,depthBuffer:!1},s=kl(t,e,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==t||this._pingPongRenderTarget.height!==e){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=kl(t,e,i);let{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=Vm(r)),this._blurMaterial=Gm(r,t,e)}return s}_compileMaterial(t){let e=new Pt(this._lodPlanes[0],t);this._renderer.compile(e,ko)}_sceneToCubeUV(t,e,i,s){let a=new Pe(90,1,e,i),c=[1,-1,1,1,1,1],l=[1,1,1,-1,-1,-1],h=this._renderer,u=h.autoClear,d=h.toneMapping;h.getClearColor(zl),h.toneMapping=Yn,h.autoClear=!1;let f=new Dr({name:"PMREM.Background",side:Be,depthWrite:!1,depthTest:!1}),g=new Pt(new ge,f),_=!1,m=t.background;m?m.isColor&&(f.color.copy(m),t.background=null,_=!0):(f.color.copy(zl),_=!0);for(let p=0;p<6;p++){let S=p%3;S===0?(a.up.set(0,c[p],0),a.lookAt(l[p],0,0)):S===1?(a.up.set(0,0,c[p]),a.lookAt(0,l[p],0)):(a.up.set(0,c[p],0),a.lookAt(0,0,l[p]));let M=this._cubeSize;gr(s,S*M,p>2?M:0,M,M),h.setRenderTarget(s),_&&h.render(g,a),h.render(t,a)}g.geometry.dispose(),g.material.dispose(),h.toneMapping=d,h.autoClear=u,t.background=m}_textureToCubeUV(t,e){let i=this._renderer,s=t.mapping===ji||t.mapping===ts;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Gl()),this._cubemapMaterial.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Vl());let r=s?this._cubemapMaterial:this._equirectMaterial,o=new Pt(this._lodPlanes[0],r),a=r.uniforms;a.envMap.value=t;let c=this._cubeSize;gr(e,0,0,3*c,2*c),i.setRenderTarget(e),i.render(o,ko)}_applyPMREM(t){let e=this._renderer,i=e.autoClear;e.autoClear=!1;let s=this._lodPlanes.length;for(let r=1;r<s;r++){let o=Math.sqrt(this._sigmas[r]*this._sigmas[r]-this._sigmas[r-1]*this._sigmas[r-1]),a=Hl[(s-r-1)%Hl.length];this._blur(t,r-1,r,o,a)}e.autoClear=i}_blur(t,e,i,s,r){let o=this._pingPongRenderTarget;this._halfBlur(t,o,e,i,s,"latitudinal",r),this._halfBlur(o,t,i,i,s,"longitudinal",r)}_halfBlur(t,e,i,s,r,o,a){let c=this._renderer,l=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");let h=3,u=new Pt(this._lodPlanes[s],l),d=l.uniforms,f=this._sizeLods[i]-1,g=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*xi-1),_=r/g,m=isFinite(r)?1+Math.floor(h*_):xi;m>xi&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${xi}`);let p=[],S=0;for(let R=0;R<xi;++R){let C=R/_,b=Math.exp(-C*C/2);p.push(b),R===0?S+=b:R<m&&(S+=2*b)}for(let R=0;R<p.length;R++)p[R]=p[R]/S;d.envMap.value=t.texture,d.samples.value=m,d.weights.value=p,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);let{_lodMax:M}=this;d.dTheta.value=g,d.mipInt.value=M-i;let x=this._sizeLods[s],D=3*x*(s>M-qi?s-M+qi:0),T=4*(this._cubeSize-x);gr(e,D,T,3*x,2*x),c.setRenderTarget(e),c.render(u,ko)}};function Vm(n){let t=[],e=[],i=[],s=n,r=n-qi+1+Bl.length;for(let o=0;o<r;o++){let a=Math.pow(2,s);e.push(a);let c=1/a;o>n-qi?c=Bl[o-n+qi-1]:o===0&&(c=0),i.push(c);let l=1/(a-2),h=-l,u=1+l,d=[h,h,u,h,u,u,h,h,u,u,h,u],f=6,g=6,_=3,m=2,p=1,S=new Float32Array(_*g*f),M=new Float32Array(m*g*f),x=new Float32Array(p*g*f);for(let T=0;T<f;T++){let R=T%3*2/3-1,C=T>2?0:-1,b=[R,C,0,R+2/3,C,0,R+2/3,C+1,0,R,C,0,R+2/3,C+1,0,R,C+1,0];S.set(b,_*g*T),M.set(d,m*g*T);let y=[T,T,T,T,T,T];x.set(y,p*g*T)}let D=new Te;D.setAttribute("position",new we(S,_)),D.setAttribute("uv",new we(M,m)),D.setAttribute("faceIndex",new we(x,p)),t.push(D),s>qi&&s--}return{lodPlanes:t,sizeLods:e,sigmas:i}}function kl(n,t,e){let i=new Ln(n,t,e);return i.texture.mapping=io,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function gr(n,t,e,i,s){n.viewport.set(t,e,i,s),n.scissor.set(t,e,i,s)}function Gm(n,t,e){let i=new Float32Array(xi),s=new L(0,1,0);return new dn({name:"SphericalGaussianBlur",defines:{n:xi,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/e,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Dc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:qn,depthTest:!1,depthWrite:!1})}function Vl(){return new dn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Dc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:qn,depthTest:!1,depthWrite:!1})}function Gl(){return new dn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Dc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:qn,depthTest:!1,depthWrite:!1})}function Dc(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function Wm(n){let t=new WeakMap,e=null;function i(a){if(a&&a.isTexture){let c=a.mapping,l=c===aa||c===ca,h=c===ji||c===ts;if(l||h){let u=t.get(a),d=u!==void 0?u.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return e===null&&(e=new zr(n)),u=l?e.fromEquirectangular(a,u):e.fromCubemap(a,u),u.texture.pmremVersion=a.pmremVersion,t.set(a,u),u.texture;if(u!==void 0)return u.texture;{let f=a.image;return l&&f&&f.height>0||h&&f&&s(f)?(e===null&&(e=new zr(n)),u=l?e.fromEquirectangular(a):e.fromCubemap(a),u.texture.pmremVersion=a.pmremVersion,t.set(a,u),a.addEventListener("dispose",r),u.texture):null}}}return a}function s(a){let c=0,l=6;for(let h=0;h<l;h++)a[h]!==void 0&&c++;return c===l}function r(a){let c=a.target;c.removeEventListener("dispose",r);let l=t.get(c);l!==void 0&&(t.delete(c),l.dispose())}function o(){t=new WeakMap,e!==null&&(e.dispose(),e=null)}return{get:i,dispose:o}}function Xm(n){let t={};function e(i){if(t[i]!==void 0)return t[i];let s;switch(i){case"WEBGL_depth_texture":s=n.getExtension("WEBGL_depth_texture")||n.getExtension("MOZ_WEBGL_depth_texture")||n.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=n.getExtension("EXT_texture_filter_anisotropic")||n.getExtension("MOZ_EXT_texture_filter_anisotropic")||n.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=n.getExtension("WEBGL_compressed_texture_s3tc")||n.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||n.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=n.getExtension("WEBGL_compressed_texture_pvrtc")||n.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=n.getExtension(i)}return t[i]=s,s}return{has:function(i){return e(i)!==null},init:function(){e("EXT_color_buffer_float"),e("WEBGL_clip_cull_distance"),e("OES_texture_float_linear"),e("EXT_color_buffer_half_float"),e("WEBGL_multisampled_render_to_texture"),e("WEBGL_render_shared_exponent")},get:function(i){let s=e(i);return s===null&&Ts("THREE.WebGLRenderer: "+i+" extension not supported."),s}}}function qm(n,t,e,i){let s={},r=new WeakMap;function o(u){let d=u.target;d.index!==null&&t.remove(d.index);for(let g in d.attributes)t.remove(d.attributes[g]);for(let g in d.morphAttributes){let _=d.morphAttributes[g];for(let m=0,p=_.length;m<p;m++)t.remove(_[m])}d.removeEventListener("dispose",o),delete s[d.id];let f=r.get(d);f&&(t.remove(f),r.delete(d)),i.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,e.memory.geometries--}function a(u,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,e.memory.geometries++),d}function c(u){let d=u.attributes;for(let g in d)t.update(d[g],n.ARRAY_BUFFER);let f=u.morphAttributes;for(let g in f){let _=f[g];for(let m=0,p=_.length;m<p;m++)t.update(_[m],n.ARRAY_BUFFER)}}function l(u){let d=[],f=u.index,g=u.attributes.position,_=0;if(f!==null){let S=f.array;_=f.version;for(let M=0,x=S.length;M<x;M+=3){let D=S[M+0],T=S[M+1],R=S[M+2];d.push(D,T,T,R,R,D)}}else if(g!==void 0){let S=g.array;_=g.version;for(let M=0,x=S.length/3-1;M<x;M+=3){let D=M+0,T=M+1,R=M+2;d.push(D,T,T,R,R,D)}}else return;let m=new(Dh(d)?Nr:Ur)(d,1);m.version=_;let p=r.get(u);p&&t.remove(p),r.set(u,m)}function h(u){let d=r.get(u);if(d){let f=u.index;f!==null&&d.version<f.version&&l(u)}else l(u);return r.get(u)}return{get:a,update:c,getWireframeAttribute:h}}function Ym(n,t,e){let i;function s(d){i=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function c(d,f){n.drawElements(i,f,r,d*o),e.update(f,i,1)}function l(d,f,g){g!==0&&(n.drawElementsInstanced(i,f,r,d*o,g),e.update(f,i,g))}function h(d,f,g){if(g===0)return;t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,f,0,r,d,0,g);let m=0;for(let p=0;p<g;p++)m+=f[p];e.update(m,i,1)}function u(d,f,g,_){if(g===0)return;let m=t.get("WEBGL_multi_draw");if(m===null)for(let p=0;p<d.length;p++)l(d[p]/o,f[p],_[p]);else{m.multiDrawElementsInstancedWEBGL(i,f,0,r,d,0,_,0,g);let p=0;for(let S=0;S<g;S++)p+=f[S]*_[S];e.update(p,i,1)}}this.setMode=s,this.setIndex=a,this.render=c,this.renderInstances=l,this.renderMultiDraw=h,this.renderMultiDrawInstances=u}function Zm(n){let t={geometries:0,textures:0},e={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,o,a){switch(e.calls++,o){case n.TRIANGLES:e.triangles+=a*(r/3);break;case n.LINES:e.lines+=a*(r/2);break;case n.LINE_STRIP:e.lines+=a*(r-1);break;case n.LINE_LOOP:e.lines+=a*r;break;case n.POINTS:e.points+=a*r;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",o);break}}function s(){e.calls=0,e.triangles=0,e.points=0,e.lines=0}return{memory:t,render:e,programs:null,autoReset:!0,reset:s,update:i}}function $m(n,t,e){let i=new WeakMap,s=new ie;function r(o,a,c){let l=o.morphTargetInfluences,h=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,u=h!==void 0?h.length:0,d=i.get(a);if(d===void 0||d.count!==u){let b=function(){R.dispose(),i.delete(a),a.removeEventListener("dispose",b)};d!==void 0&&d.texture.dispose();let f=a.morphAttributes.position!==void 0,g=a.morphAttributes.normal!==void 0,_=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],p=a.morphAttributes.normal||[],S=a.morphAttributes.color||[],M=0;f===!0&&(M=1),g===!0&&(M=2),_===!0&&(M=3);let x=a.attributes.position.count*M,D=1;x>t.maxTextureSize&&(D=Math.ceil(x/t.maxTextureSize),x=t.maxTextureSize);let T=new Float32Array(x*D*4*u),R=new Pr(T,x,D,u);R.type=Rn,R.needsUpdate=!0;let C=M*4;for(let y=0;y<u;y++){let I=m[y],B=p[y],N=S[y],H=x*D*4*y;for(let Y=0;Y<I.count;Y++){let V=Y*C;f===!0&&(s.fromBufferAttribute(I,Y),T[H+V+0]=s.x,T[H+V+1]=s.y,T[H+V+2]=s.z,T[H+V+3]=0),g===!0&&(s.fromBufferAttribute(B,Y),T[H+V+4]=s.x,T[H+V+5]=s.y,T[H+V+6]=s.z,T[H+V+7]=0),_===!0&&(s.fromBufferAttribute(N,Y),T[H+V+8]=s.x,T[H+V+9]=s.y,T[H+V+10]=s.z,T[H+V+11]=N.itemSize===4?s.w:1)}}d={count:u,texture:R,size:new j(x,D)},i.set(a,d),a.addEventListener("dispose",b)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)c.getUniforms().setValue(n,"morphTexture",o.morphTexture,e);else{let f=0;for(let _=0;_<l.length;_++)f+=l[_];let g=a.morphTargetsRelative?1:1-f;c.getUniforms().setValue(n,"morphTargetBaseInfluence",g),c.getUniforms().setValue(n,"morphTargetInfluences",l)}c.getUniforms().setValue(n,"morphTargetsTexture",d.texture,e),c.getUniforms().setValue(n,"morphTargetsTextureSize",d.size)}return{update:r}}function Jm(n,t,e,i){let s=new WeakMap;function r(c){let l=i.render.frame,h=c.geometry,u=t.get(c,h);if(s.get(u)!==l&&(t.update(u),s.set(u,l)),c.isInstancedMesh&&(c.hasEventListener("dispose",a)===!1&&c.addEventListener("dispose",a),s.get(c)!==l&&(e.update(c.instanceMatrix,n.ARRAY_BUFFER),c.instanceColor!==null&&e.update(c.instanceColor,n.ARRAY_BUFFER),s.set(c,l))),c.isSkinnedMesh){let d=c.skeleton;s.get(d)!==l&&(d.update(),s.set(d,l))}return u}function o(){s=new WeakMap}function a(c){let l=c.target;l.removeEventListener("dispose",a),e.remove(l.instanceMatrix),l.instanceColor!==null&&e.remove(l.instanceColor)}return{update:r,dispose:o}}var Hr=class extends We{constructor(t,e,i,s,r,o,a,c,l,h=$i){if(h!==$i&&h!==ns)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");i===void 0&&h===$i&&(i=Mi),i===void 0&&h===ns&&(i=es),super(null,s,r,o,a,c,h,i,l),this.isDepthTexture=!0,this.image={width:t,height:e},this.magFilter=a!==void 0?a:rn,this.minFilter=c!==void 0?c:rn,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(t){return super.copy(t),this.compareFunction=t.compareFunction,this}toJSON(t){let e=super.toJSON(t);return this.compareFunction!==null&&(e.compareFunction=this.compareFunction),e}},Oh=new We,Wl=new Hr(1,1),Bh=new Pr,zh=new za,Hh=new Or,Xl=[],ql=[],Yl=new Float32Array(16),Zl=new Float32Array(9),$l=new Float32Array(4);function ls(n,t,e){let i=n[0];if(i<=0||i>0)return n;let s=t*e,r=Xl[s];if(r===void 0&&(r=new Float32Array(s),Xl[s]=r),t!==0){i.toArray(r,0);for(let o=1,a=0;o!==t;++o)a+=e,n[o].toArray(r,a)}return r}function Me(n,t){if(n.length!==t.length)return!1;for(let e=0,i=n.length;e<i;e++)if(n[e]!==t[e])return!1;return!0}function Se(n,t){for(let e=0,i=t.length;e<i;e++)n[e]=t[e]}function oo(n,t){let e=ql[t];e===void 0&&(e=new Int32Array(t),ql[t]=e);for(let i=0;i!==t;++i)e[i]=n.allocateTextureUnit();return e}function Km(n,t){let e=this.cache;e[0]!==t&&(n.uniform1f(this.addr,t),e[0]=t)}function Qm(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(n.uniform2f(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Me(e,t))return;n.uniform2fv(this.addr,t),Se(e,t)}}function jm(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(n.uniform3f(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else if(t.r!==void 0)(e[0]!==t.r||e[1]!==t.g||e[2]!==t.b)&&(n.uniform3f(this.addr,t.r,t.g,t.b),e[0]=t.r,e[1]=t.g,e[2]=t.b);else{if(Me(e,t))return;n.uniform3fv(this.addr,t),Se(e,t)}}function tg(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(n.uniform4f(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Me(e,t))return;n.uniform4fv(this.addr,t),Se(e,t)}}function eg(n,t){let e=this.cache,i=t.elements;if(i===void 0){if(Me(e,t))return;n.uniformMatrix2fv(this.addr,!1,t),Se(e,t)}else{if(Me(e,i))return;$l.set(i),n.uniformMatrix2fv(this.addr,!1,$l),Se(e,i)}}function ng(n,t){let e=this.cache,i=t.elements;if(i===void 0){if(Me(e,t))return;n.uniformMatrix3fv(this.addr,!1,t),Se(e,t)}else{if(Me(e,i))return;Zl.set(i),n.uniformMatrix3fv(this.addr,!1,Zl),Se(e,i)}}function ig(n,t){let e=this.cache,i=t.elements;if(i===void 0){if(Me(e,t))return;n.uniformMatrix4fv(this.addr,!1,t),Se(e,t)}else{if(Me(e,i))return;Yl.set(i),n.uniformMatrix4fv(this.addr,!1,Yl),Se(e,i)}}function sg(n,t){let e=this.cache;e[0]!==t&&(n.uniform1i(this.addr,t),e[0]=t)}function rg(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(n.uniform2i(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Me(e,t))return;n.uniform2iv(this.addr,t),Se(e,t)}}function og(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(n.uniform3i(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Me(e,t))return;n.uniform3iv(this.addr,t),Se(e,t)}}function ag(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(n.uniform4i(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Me(e,t))return;n.uniform4iv(this.addr,t),Se(e,t)}}function cg(n,t){let e=this.cache;e[0]!==t&&(n.uniform1ui(this.addr,t),e[0]=t)}function lg(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(n.uniform2ui(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Me(e,t))return;n.uniform2uiv(this.addr,t),Se(e,t)}}function hg(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(n.uniform3ui(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Me(e,t))return;n.uniform3uiv(this.addr,t),Se(e,t)}}function ug(n,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(n.uniform4ui(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Me(e,t))return;n.uniform4uiv(this.addr,t),Se(e,t)}}function dg(n,t,e){let i=this.cache,s=e.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s);let r;this.type===n.SAMPLER_2D_SHADOW?(Wl.compareFunction=Lh,r=Wl):r=Oh,e.setTexture2D(t||r,s)}function fg(n,t,e){let i=this.cache,s=e.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),e.setTexture3D(t||zh,s)}function pg(n,t,e){let i=this.cache,s=e.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),e.setTextureCube(t||Hh,s)}function mg(n,t,e){let i=this.cache,s=e.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),e.setTexture2DArray(t||Bh,s)}function gg(n){switch(n){case 5126:return Km;case 35664:return Qm;case 35665:return jm;case 35666:return tg;case 35674:return eg;case 35675:return ng;case 35676:return ig;case 5124:case 35670:return sg;case 35667:case 35671:return rg;case 35668:case 35672:return og;case 35669:case 35673:return ag;case 5125:return cg;case 36294:return lg;case 36295:return hg;case 36296:return ug;case 35678:case 36198:case 36298:case 36306:case 35682:return dg;case 35679:case 36299:case 36307:return fg;case 35680:case 36300:case 36308:case 36293:return pg;case 36289:case 36303:case 36311:case 36292:return mg}}function _g(n,t){n.uniform1fv(this.addr,t)}function xg(n,t){let e=ls(t,this.size,2);n.uniform2fv(this.addr,e)}function yg(n,t){let e=ls(t,this.size,3);n.uniform3fv(this.addr,e)}function vg(n,t){let e=ls(t,this.size,4);n.uniform4fv(this.addr,e)}function Mg(n,t){let e=ls(t,this.size,4);n.uniformMatrix2fv(this.addr,!1,e)}function Sg(n,t){let e=ls(t,this.size,9);n.uniformMatrix3fv(this.addr,!1,e)}function Eg(n,t){let e=ls(t,this.size,16);n.uniformMatrix4fv(this.addr,!1,e)}function bg(n,t){n.uniform1iv(this.addr,t)}function wg(n,t){n.uniform2iv(this.addr,t)}function Tg(n,t){n.uniform3iv(this.addr,t)}function Ag(n,t){n.uniform4iv(this.addr,t)}function Rg(n,t){n.uniform1uiv(this.addr,t)}function Cg(n,t){n.uniform2uiv(this.addr,t)}function Ig(n,t){n.uniform3uiv(this.addr,t)}function Pg(n,t){n.uniform4uiv(this.addr,t)}function Lg(n,t,e){let i=this.cache,s=t.length,r=oo(e,s);Me(i,r)||(n.uniform1iv(this.addr,r),Se(i,r));for(let o=0;o!==s;++o)e.setTexture2D(t[o]||Oh,r[o])}function Dg(n,t,e){let i=this.cache,s=t.length,r=oo(e,s);Me(i,r)||(n.uniform1iv(this.addr,r),Se(i,r));for(let o=0;o!==s;++o)e.setTexture3D(t[o]||zh,r[o])}function Ug(n,t,e){let i=this.cache,s=t.length,r=oo(e,s);Me(i,r)||(n.uniform1iv(this.addr,r),Se(i,r));for(let o=0;o!==s;++o)e.setTextureCube(t[o]||Hh,r[o])}function Ng(n,t,e){let i=this.cache,s=t.length,r=oo(e,s);Me(i,r)||(n.uniform1iv(this.addr,r),Se(i,r));for(let o=0;o!==s;++o)e.setTexture2DArray(t[o]||Bh,r[o])}function Fg(n){switch(n){case 5126:return _g;case 35664:return xg;case 35665:return yg;case 35666:return vg;case 35674:return Mg;case 35675:return Sg;case 35676:return Eg;case 5124:case 35670:return bg;case 35667:case 35671:return wg;case 35668:case 35672:return Tg;case 35669:case 35673:return Ag;case 5125:return Rg;case 36294:return Cg;case 36295:return Ig;case 36296:return Pg;case 35678:case 36198:case 36298:case 36306:case 35682:return Lg;case 35679:case 36299:case 36307:return Dg;case 35680:case 36300:case 36308:case 36293:return Ug;case 36289:case 36303:case 36311:case 36292:return Ng}}var Va=class{constructor(t,e,i){this.id=t,this.addr=i,this.cache=[],this.type=e.type,this.setValue=gg(e.type)}},Ga=class{constructor(t,e,i){this.id=t,this.addr=i,this.cache=[],this.type=e.type,this.size=e.size,this.setValue=Fg(e.type)}},Wa=class{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,i){let s=this.seq;for(let r=0,o=s.length;r!==o;++r){let a=s[r];a.setValue(t,e[a.id],i)}}},qo=/(\w+)(\])?(\[|\.)?/g;function Jl(n,t){n.seq.push(t),n.map[t.id]=t}function Og(n,t,e){let i=n.name,s=i.length;for(qo.lastIndex=0;;){let r=qo.exec(i),o=qo.lastIndex,a=r[1],c=r[2]==="]",l=r[3];if(c&&(a=a|0),l===void 0||l==="["&&o+2===s){Jl(e,l===void 0?new Va(a,n,t):new Ga(a,n,t));break}else{let u=e.map[a];u===void 0&&(u=new Wa(a),Jl(e,u)),e=u}}}var Ki=class{constructor(t,e){this.seq=[],this.map={};let i=t.getProgramParameter(e,t.ACTIVE_UNIFORMS);for(let s=0;s<i;++s){let r=t.getActiveUniform(e,s),o=t.getUniformLocation(e,r.name);Og(r,o,this)}}setValue(t,e,i,s){let r=this.map[e];r!==void 0&&r.setValue(t,i,s)}setOptional(t,e,i){let s=e[i];s!==void 0&&this.setValue(t,i,s)}static upload(t,e,i,s){for(let r=0,o=e.length;r!==o;++r){let a=e[r],c=i[a.id];c.needsUpdate!==!1&&a.setValue(t,c.value,s)}}static seqWithValue(t,e){let i=[];for(let s=0,r=t.length;s!==r;++s){let o=t[s];o.id in e&&i.push(o)}return i}};function Kl(n,t,e){let i=n.createShader(t);return n.shaderSource(i,e),n.compileShader(i),i}var Bg=37297,zg=0;function Hg(n,t){let e=n.split(`
`),i=[],s=Math.max(t-6,0),r=Math.min(t+6,e.length);for(let o=s;o<r;o++){let a=o+1;i.push(`${a===t?">":" "} ${a}: ${e[o]}`)}return i.join(`
`)}var Ql=new Wt;function kg(n){Jt._getMatrix(Ql,Jt.workingColorSpace,n);let t=`mat3( ${Ql.elements.map(e=>e.toFixed(4))} )`;switch(Jt.getTransfer(n)){case so:return[t,"LinearTransferOETF"];case ne:return[t,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space: ",n),[t,"LinearTransferOETF"]}}function jl(n,t,e){let i=n.getShaderParameter(t,n.COMPILE_STATUS),s=n.getShaderInfoLog(t).trim();if(i&&s==="")return"";let r=/ERROR: 0:(\d+)/.exec(s);if(r){let o=parseInt(r[1]);return e.toUpperCase()+`

`+s+`

`+Hg(n.getShaderSource(t),o)}else return s}function Vg(n,t){let e=kg(t);return[`vec4 ${n}( vec4 value ) {`,`	return ${e[1]}( vec4( value.rgb * ${e[0]}, value.a ) );`,"}"].join(`
`)}function Gg(n,t){let e;switch(t){case ju:e="Linear";break;case td:e="Reinhard";break;case ed:e="Cineon";break;case wc:e="ACESFilmic";break;case id:e="AgX";break;case sd:e="Neutral";break;case nd:e="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",t),e="Linear"}return"vec3 "+n+"( vec3 color ) { return "+e+"ToneMapping( color ); }"}var _r=new L;function Wg(){Jt.getLuminanceCoefficients(_r);let n=_r.x.toFixed(4),t=_r.y.toFixed(4),e=_r.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${n}, ${t}, ${e} );`,"	return dot( weights, rgb );","}"].join(`
`)}function Xg(n){return[n.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",n.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(As).join(`
`)}function qg(n){let t=[];for(let e in n){let i=n[e];i!==!1&&t.push("#define "+e+" "+i)}return t.join(`
`)}function Yg(n,t){let e={},i=n.getProgramParameter(t,n.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=n.getActiveAttrib(t,s),o=r.name,a=1;r.type===n.FLOAT_MAT2&&(a=2),r.type===n.FLOAT_MAT3&&(a=3),r.type===n.FLOAT_MAT4&&(a=4),e[o]={type:r.type,location:n.getAttribLocation(t,o),locationSize:a}}return e}function As(n){return n!==""}function th(n,t){let e=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return n.replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,e).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows)}function eh(n,t){return n.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}var Zg=/^[ \t]*#include +<([\w\d./]+)>/gm;function Xa(n){return n.replace(Zg,Jg)}var $g=new Map;function Jg(n,t){let e=qt[t];if(e===void 0){let i=$g.get(t);if(i!==void 0)e=qt[i],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',t,i);else throw new Error("Can not resolve #include <"+t+">")}return Xa(e)}var Kg=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function nh(n){return n.replace(Kg,Qg)}function Qg(n,t,e,i){let s="";for(let r=parseInt(t);r<parseInt(e);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function ih(n){let t=`precision ${n.precision} float;
	precision ${n.precision} int;
	precision ${n.precision} sampler2D;
	precision ${n.precision} samplerCube;
	precision ${n.precision} sampler3D;
	precision ${n.precision} sampler2DArray;
	precision ${n.precision} sampler2DShadow;
	precision ${n.precision} samplerCubeShadow;
	precision ${n.precision} sampler2DArrayShadow;
	precision ${n.precision} isampler2D;
	precision ${n.precision} isampler3D;
	precision ${n.precision} isamplerCube;
	precision ${n.precision} isampler2DArray;
	precision ${n.precision} usampler2D;
	precision ${n.precision} usampler3D;
	precision ${n.precision} usamplerCube;
	precision ${n.precision} usampler2DArray;
	`;return n.precision==="highp"?t+=`
#define HIGH_PRECISION`:n.precision==="mediump"?t+=`
#define MEDIUM_PRECISION`:n.precision==="lowp"&&(t+=`
#define LOW_PRECISION`),t}function jg(n){let t="SHADOWMAP_TYPE_BASIC";return n.shadowMapType===xh?t="SHADOWMAP_TYPE_PCF":n.shadowMapType===bc?t="SHADOWMAP_TYPE_PCF_SOFT":n.shadowMapType===Tn&&(t="SHADOWMAP_TYPE_VSM"),t}function t0(n){let t="ENVMAP_TYPE_CUBE";if(n.envMap)switch(n.envMapMode){case ji:case ts:t="ENVMAP_TYPE_CUBE";break;case io:t="ENVMAP_TYPE_CUBE_UV";break}return t}function e0(n){let t="ENVMAP_MODE_REFLECTION";if(n.envMap)switch(n.envMapMode){case ts:t="ENVMAP_MODE_REFRACTION";break}return t}function n0(n){let t="ENVMAP_BLENDING_NONE";if(n.envMap)switch(n.combine){case yh:t="ENVMAP_BLENDING_MULTIPLY";break;case Ku:t="ENVMAP_BLENDING_MIX";break;case Qu:t="ENVMAP_BLENDING_ADD";break}return t}function i0(n){let t=n.envMapCubeUVHeight;if(t===null)return null;let e=Math.log2(t)-2,i=1/t;return{texelWidth:1/(3*Math.max(Math.pow(2,e),7*16)),texelHeight:i,maxMip:e}}function s0(n,t,e,i){let s=n.getContext(),r=e.defines,o=e.vertexShader,a=e.fragmentShader,c=jg(e),l=t0(e),h=e0(e),u=n0(e),d=i0(e),f=Xg(e),g=qg(r),_=s.createProgram(),m,p,S=e.glslVersion?"#version "+e.glslVersion+`
`:"";e.isRawShaderMaterial?(m=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter(As).join(`
`),m.length>0&&(m+=`
`),p=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter(As).join(`
`),p.length>0&&(p+=`
`)):(m=[ih(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",e.batching?"#define USE_BATCHING":"",e.batchingColor?"#define USE_BATCHING_COLOR":"",e.instancing?"#define USE_INSTANCING":"",e.instancingColor?"#define USE_INSTANCING_COLOR":"",e.instancingMorph?"#define USE_INSTANCING_MORPH":"",e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+h:"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.displacementMap?"#define USE_DISPLACEMENTMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.mapUv?"#define MAP_UV "+e.mapUv:"",e.alphaMapUv?"#define ALPHAMAP_UV "+e.alphaMapUv:"",e.lightMapUv?"#define LIGHTMAP_UV "+e.lightMapUv:"",e.aoMapUv?"#define AOMAP_UV "+e.aoMapUv:"",e.emissiveMapUv?"#define EMISSIVEMAP_UV "+e.emissiveMapUv:"",e.bumpMapUv?"#define BUMPMAP_UV "+e.bumpMapUv:"",e.normalMapUv?"#define NORMALMAP_UV "+e.normalMapUv:"",e.displacementMapUv?"#define DISPLACEMENTMAP_UV "+e.displacementMapUv:"",e.metalnessMapUv?"#define METALNESSMAP_UV "+e.metalnessMapUv:"",e.roughnessMapUv?"#define ROUGHNESSMAP_UV "+e.roughnessMapUv:"",e.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+e.anisotropyMapUv:"",e.clearcoatMapUv?"#define CLEARCOATMAP_UV "+e.clearcoatMapUv:"",e.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+e.clearcoatNormalMapUv:"",e.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+e.clearcoatRoughnessMapUv:"",e.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+e.iridescenceMapUv:"",e.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+e.iridescenceThicknessMapUv:"",e.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+e.sheenColorMapUv:"",e.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+e.sheenRoughnessMapUv:"",e.specularMapUv?"#define SPECULARMAP_UV "+e.specularMapUv:"",e.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+e.specularColorMapUv:"",e.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+e.specularIntensityMapUv:"",e.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+e.transmissionMapUv:"",e.thicknessMapUv?"#define THICKNESSMAP_UV "+e.thicknessMapUv:"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.flatShading?"#define FLAT_SHADED":"",e.skinning?"#define USE_SKINNING":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals&&e.flatShading===!1?"#define USE_MORPHNORMALS":"",e.morphColors?"#define USE_MORPHCOLORS":"",e.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+e.morphTextureStride:"",e.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+e.morphTargetsCount:"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(As).join(`
`),p=[ih(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",e.map?"#define USE_MAP":"",e.matcap?"#define USE_MATCAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+l:"",e.envMap?"#define "+h:"",e.envMap?"#define "+u:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoat?"#define USE_CLEARCOAT":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.dispersion?"#define USE_DISPERSION":"",e.iridescence?"#define USE_IRIDESCENCE":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaTest?"#define USE_ALPHATEST":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.sheen?"#define USE_SHEEN":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors||e.instancingColor||e.batchingColor?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.gradientMap?"#define USE_GRADIENTMAP":"",e.flatShading?"#define FLAT_SHADED":"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",e.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",e.toneMapping!==Yn?"#define TONE_MAPPING":"",e.toneMapping!==Yn?qt.tonemapping_pars_fragment:"",e.toneMapping!==Yn?Gg("toneMapping",e.toneMapping):"",e.dithering?"#define DITHERING":"",e.opaque?"#define OPAQUE":"",qt.colorspace_pars_fragment,Vg("linearToOutputTexel",e.outputColorSpace),Wg(),e.useDepthPacking?"#define DEPTH_PACKING "+e.depthPacking:"",`
`].filter(As).join(`
`)),o=Xa(o),o=th(o,e),o=eh(o,e),a=Xa(a),a=th(a,e),a=eh(a,e),o=nh(o),a=nh(a),e.isRawShaderMaterial!==!0&&(S=`#version 300 es
`,m=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,p=["#define varying in",e.glslVersion===gl?"":"layout(location = 0) out highp vec4 pc_fragColor;",e.glslVersion===gl?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+p);let M=S+m+o,x=S+p+a,D=Kl(s,s.VERTEX_SHADER,M),T=Kl(s,s.FRAGMENT_SHADER,x);s.attachShader(_,D),s.attachShader(_,T),e.index0AttributeName!==void 0?s.bindAttribLocation(_,0,e.index0AttributeName):e.morphTargets===!0&&s.bindAttribLocation(_,0,"position"),s.linkProgram(_);function R(I){if(n.debug.checkShaderErrors){let B=s.getProgramInfoLog(_).trim(),N=s.getShaderInfoLog(D).trim(),H=s.getShaderInfoLog(T).trim(),Y=!0,V=!0;if(s.getProgramParameter(_,s.LINK_STATUS)===!1)if(Y=!1,typeof n.debug.onShaderError=="function")n.debug.onShaderError(s,_,D,T);else{let rt=jl(s,D,"vertex"),X=jl(s,T,"fragment");console.error("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(_,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+B+`
`+rt+`
`+X)}else B!==""?console.warn("THREE.WebGLProgram: Program Info Log:",B):(N===""||H==="")&&(V=!1);V&&(I.diagnostics={runnable:Y,programLog:B,vertexShader:{log:N,prefix:m},fragmentShader:{log:H,prefix:p}})}s.deleteShader(D),s.deleteShader(T),C=new Ki(s,_),b=Yg(s,_)}let C;this.getUniforms=function(){return C===void 0&&R(this),C};let b;this.getAttributes=function(){return b===void 0&&R(this),b};let y=e.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return y===!1&&(y=s.getProgramParameter(_,Bg)),y},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(_),this.program=void 0},this.type=e.shaderType,this.name=e.shaderName,this.id=zg++,this.cacheKey=t,this.usedTimes=1,this.program=_,this.vertexShader=D,this.fragmentShader=T,this}var r0=0,qa=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t){let e=t.vertexShader,i=t.fragmentShader,s=this._getShaderStage(e),r=this._getShaderStage(i),o=this._getShaderCacheForMaterial(t);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(t){let e=this.materialCache.get(t);for(let i of e)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(t),this}getVertexShaderID(t){return this._getShaderStage(t.vertexShader).id}getFragmentShaderID(t){return this._getShaderStage(t.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){let e=this.materialCache,i=e.get(t);return i===void 0&&(i=new Set,e.set(t,i)),i}_getShaderStage(t){let e=this.shaderCache,i=e.get(t);return i===void 0&&(i=new Ya(t),e.set(t,i)),i}},Ya=class{constructor(t){this.id=r0++,this.code=t,this.usedTimes=0}};function o0(n,t,e,i,s,r,o){let a=new Os,c=new qa,l=new Set,h=[],u=s.logarithmicDepthBuffer,d=s.vertexTextures,f=s.precision,g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(b){return l.add(b),b===0?"uv":`uv${b}`}function m(b,y,I,B,N){let H=B.fog,Y=N.geometry,V=b.isMeshStandardMaterial?B.environment:null,rt=(b.isMeshStandardMaterial?e:t).get(b.envMap||V),X=rt&&rt.mapping===io?rt.image.height:null,it=g[b.type];b.precision!==null&&(f=s.getMaxPrecision(b.precision),f!==b.precision&&console.warn("THREE.WebGLProgram.getParameters:",b.precision,"not supported, using",f,"instead."));let dt=Y.morphAttributes.position||Y.morphAttributes.normal||Y.morphAttributes.color,vt=dt!==void 0?dt.length:0,zt=0;Y.morphAttributes.position!==void 0&&(zt=1),Y.morphAttributes.normal!==void 0&&(zt=2),Y.morphAttributes.color!==void 0&&(zt=3);let Kt,P,k,nt;if(it){let ee=ln[it];Kt=ee.vertexShader,P=ee.fragmentShader}else Kt=b.vertexShader,P=b.fragmentShader,c.update(b),k=c.getVertexShaderID(b),nt=c.getFragmentShaderID(b);let tt=n.getRenderTarget(),wt=n.state.buffers.depth.getReversed(),Dt=N.isInstancedMesh===!0,Lt=N.isBatchedMesh===!0,Yt=!!b.map,Z=!!b.matcap,st=!!rt,A=!!b.aoMap,St=!!b.lightMap,ot=!!b.bumpMap,Mt=!!b.normalMap,ht=!!b.displacementMap,Ut=!!b.emissiveMap,xt=!!b.metalnessMap,w=!!b.roughnessMap,v=b.anisotropy>0,z=b.clearcoat>0,$=b.dispersion>0,et=b.iridescence>0,J=b.sheen>0,Tt=b.transmission>0,ft=v&&!!b.anisotropyMap,yt=z&&!!b.clearcoatMap,Zt=z&&!!b.clearcoatNormalMap,at=z&&!!b.clearcoatRoughnessMap,Et=et&&!!b.iridescenceMap,Nt=et&&!!b.iridescenceThicknessMap,Bt=J&&!!b.sheenColorMap,bt=J&&!!b.sheenRoughnessMap,$t=!!b.specularMap,Xt=!!b.specularColorMap,oe=!!b.specularIntensityMap,U=Tt&&!!b.transmissionMap,pt=Tt&&!!b.thicknessMap,q=!!b.gradientMap,Q=!!b.alphaMap,_t=b.alphaTest>0,mt=!!b.alphaHash,Vt=!!b.extensions,me=Yn;b.toneMapped&&(tt===null||tt.isXRRenderTarget===!0)&&(me=n.toneMapping);let Re={shaderID:it,shaderType:b.type,shaderName:b.name,vertexShader:Kt,fragmentShader:P,defines:b.defines,customVertexShaderID:k,customFragmentShaderID:nt,isRawShaderMaterial:b.isRawShaderMaterial===!0,glslVersion:b.glslVersion,precision:f,batching:Lt,batchingColor:Lt&&N._colorsTexture!==null,instancing:Dt,instancingColor:Dt&&N.instanceColor!==null,instancingMorph:Dt&&N.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:tt===null?n.outputColorSpace:tt.isXRRenderTarget===!0?tt.texture.colorSpace:cs,alphaToCoverage:!!b.alphaToCoverage,map:Yt,matcap:Z,envMap:st,envMapMode:st&&rt.mapping,envMapCubeUVHeight:X,aoMap:A,lightMap:St,bumpMap:ot,normalMap:Mt,displacementMap:d&&ht,emissiveMap:Ut,normalMapObjectSpace:Mt&&b.normalMapType===cd,normalMapTangentSpace:Mt&&b.normalMapType===Ph,metalnessMap:xt,roughnessMap:w,anisotropy:v,anisotropyMap:ft,clearcoat:z,clearcoatMap:yt,clearcoatNormalMap:Zt,clearcoatRoughnessMap:at,dispersion:$,iridescence:et,iridescenceMap:Et,iridescenceThicknessMap:Nt,sheen:J,sheenColorMap:Bt,sheenRoughnessMap:bt,specularMap:$t,specularColorMap:Xt,specularIntensityMap:oe,transmission:Tt,transmissionMap:U,thicknessMap:pt,gradientMap:q,opaque:b.transparent===!1&&b.blending===Zi&&b.alphaToCoverage===!1,alphaMap:Q,alphaTest:_t,alphaHash:mt,combine:b.combine,mapUv:Yt&&_(b.map.channel),aoMapUv:A&&_(b.aoMap.channel),lightMapUv:St&&_(b.lightMap.channel),bumpMapUv:ot&&_(b.bumpMap.channel),normalMapUv:Mt&&_(b.normalMap.channel),displacementMapUv:ht&&_(b.displacementMap.channel),emissiveMapUv:Ut&&_(b.emissiveMap.channel),metalnessMapUv:xt&&_(b.metalnessMap.channel),roughnessMapUv:w&&_(b.roughnessMap.channel),anisotropyMapUv:ft&&_(b.anisotropyMap.channel),clearcoatMapUv:yt&&_(b.clearcoatMap.channel),clearcoatNormalMapUv:Zt&&_(b.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:at&&_(b.clearcoatRoughnessMap.channel),iridescenceMapUv:Et&&_(b.iridescenceMap.channel),iridescenceThicknessMapUv:Nt&&_(b.iridescenceThicknessMap.channel),sheenColorMapUv:Bt&&_(b.sheenColorMap.channel),sheenRoughnessMapUv:bt&&_(b.sheenRoughnessMap.channel),specularMapUv:$t&&_(b.specularMap.channel),specularColorMapUv:Xt&&_(b.specularColorMap.channel),specularIntensityMapUv:oe&&_(b.specularIntensityMap.channel),transmissionMapUv:U&&_(b.transmissionMap.channel),thicknessMapUv:pt&&_(b.thicknessMap.channel),alphaMapUv:Q&&_(b.alphaMap.channel),vertexTangents:!!Y.attributes.tangent&&(Mt||v),vertexColors:b.vertexColors,vertexAlphas:b.vertexColors===!0&&!!Y.attributes.color&&Y.attributes.color.itemSize===4,pointsUvs:N.isPoints===!0&&!!Y.attributes.uv&&(Yt||Q),fog:!!H,useFog:b.fog===!0,fogExp2:!!H&&H.isFogExp2,flatShading:b.flatShading===!0,sizeAttenuation:b.sizeAttenuation===!0,logarithmicDepthBuffer:u,reverseDepthBuffer:wt,skinning:N.isSkinnedMesh===!0,morphTargets:Y.morphAttributes.position!==void 0,morphNormals:Y.morphAttributes.normal!==void 0,morphColors:Y.morphAttributes.color!==void 0,morphTargetsCount:vt,morphTextureStride:zt,numDirLights:y.directional.length,numPointLights:y.point.length,numSpotLights:y.spot.length,numSpotLightMaps:y.spotLightMap.length,numRectAreaLights:y.rectArea.length,numHemiLights:y.hemi.length,numDirLightShadows:y.directionalShadowMap.length,numPointLightShadows:y.pointShadowMap.length,numSpotLightShadows:y.spotShadowMap.length,numSpotLightShadowsWithMaps:y.numSpotLightShadowsWithMaps,numLightProbes:y.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:b.dithering,shadowMapEnabled:n.shadowMap.enabled&&I.length>0,shadowMapType:n.shadowMap.type,toneMapping:me,decodeVideoTexture:Yt&&b.map.isVideoTexture===!0&&Jt.getTransfer(b.map.colorSpace)===ne,decodeVideoTextureEmissive:Ut&&b.emissiveMap.isVideoTexture===!0&&Jt.getTransfer(b.emissiveMap.colorSpace)===ne,premultipliedAlpha:b.premultipliedAlpha,doubleSided:b.side===ve,flipSided:b.side===Be,useDepthPacking:b.depthPacking>=0,depthPacking:b.depthPacking||0,index0AttributeName:b.index0AttributeName,extensionClipCullDistance:Vt&&b.extensions.clipCullDistance===!0&&i.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Vt&&b.extensions.multiDraw===!0||Lt)&&i.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:i.has("KHR_parallel_shader_compile"),customProgramCacheKey:b.customProgramCacheKey()};return Re.vertexUv1s=l.has(1),Re.vertexUv2s=l.has(2),Re.vertexUv3s=l.has(3),l.clear(),Re}function p(b){let y=[];if(b.shaderID?y.push(b.shaderID):(y.push(b.customVertexShaderID),y.push(b.customFragmentShaderID)),b.defines!==void 0)for(let I in b.defines)y.push(I),y.push(b.defines[I]);return b.isRawShaderMaterial===!1&&(S(y,b),M(y,b),y.push(n.outputColorSpace)),y.push(b.customProgramCacheKey),y.join()}function S(b,y){b.push(y.precision),b.push(y.outputColorSpace),b.push(y.envMapMode),b.push(y.envMapCubeUVHeight),b.push(y.mapUv),b.push(y.alphaMapUv),b.push(y.lightMapUv),b.push(y.aoMapUv),b.push(y.bumpMapUv),b.push(y.normalMapUv),b.push(y.displacementMapUv),b.push(y.emissiveMapUv),b.push(y.metalnessMapUv),b.push(y.roughnessMapUv),b.push(y.anisotropyMapUv),b.push(y.clearcoatMapUv),b.push(y.clearcoatNormalMapUv),b.push(y.clearcoatRoughnessMapUv),b.push(y.iridescenceMapUv),b.push(y.iridescenceThicknessMapUv),b.push(y.sheenColorMapUv),b.push(y.sheenRoughnessMapUv),b.push(y.specularMapUv),b.push(y.specularColorMapUv),b.push(y.specularIntensityMapUv),b.push(y.transmissionMapUv),b.push(y.thicknessMapUv),b.push(y.combine),b.push(y.fogExp2),b.push(y.sizeAttenuation),b.push(y.morphTargetsCount),b.push(y.morphAttributeCount),b.push(y.numDirLights),b.push(y.numPointLights),b.push(y.numSpotLights),b.push(y.numSpotLightMaps),b.push(y.numHemiLights),b.push(y.numRectAreaLights),b.push(y.numDirLightShadows),b.push(y.numPointLightShadows),b.push(y.numSpotLightShadows),b.push(y.numSpotLightShadowsWithMaps),b.push(y.numLightProbes),b.push(y.shadowMapType),b.push(y.toneMapping),b.push(y.numClippingPlanes),b.push(y.numClipIntersection),b.push(y.depthPacking)}function M(b,y){a.disableAll(),y.supportsVertexTextures&&a.enable(0),y.instancing&&a.enable(1),y.instancingColor&&a.enable(2),y.instancingMorph&&a.enable(3),y.matcap&&a.enable(4),y.envMap&&a.enable(5),y.normalMapObjectSpace&&a.enable(6),y.normalMapTangentSpace&&a.enable(7),y.clearcoat&&a.enable(8),y.iridescence&&a.enable(9),y.alphaTest&&a.enable(10),y.vertexColors&&a.enable(11),y.vertexAlphas&&a.enable(12),y.vertexUv1s&&a.enable(13),y.vertexUv2s&&a.enable(14),y.vertexUv3s&&a.enable(15),y.vertexTangents&&a.enable(16),y.anisotropy&&a.enable(17),y.alphaHash&&a.enable(18),y.batching&&a.enable(19),y.dispersion&&a.enable(20),y.batchingColor&&a.enable(21),b.push(a.mask),a.disableAll(),y.fog&&a.enable(0),y.useFog&&a.enable(1),y.flatShading&&a.enable(2),y.logarithmicDepthBuffer&&a.enable(3),y.reverseDepthBuffer&&a.enable(4),y.skinning&&a.enable(5),y.morphTargets&&a.enable(6),y.morphNormals&&a.enable(7),y.morphColors&&a.enable(8),y.premultipliedAlpha&&a.enable(9),y.shadowMapEnabled&&a.enable(10),y.doubleSided&&a.enable(11),y.flipSided&&a.enable(12),y.useDepthPacking&&a.enable(13),y.dithering&&a.enable(14),y.transmission&&a.enable(15),y.sheen&&a.enable(16),y.opaque&&a.enable(17),y.pointsUvs&&a.enable(18),y.decodeVideoTexture&&a.enable(19),y.decodeVideoTextureEmissive&&a.enable(20),y.alphaToCoverage&&a.enable(21),b.push(a.mask)}function x(b){let y=g[b.type],I;if(y){let B=ln[y];I=$d.clone(B.uniforms)}else I=b.uniforms;return I}function D(b,y){let I;for(let B=0,N=h.length;B<N;B++){let H=h[B];if(H.cacheKey===y){I=H,++I.usedTimes;break}}return I===void 0&&(I=new s0(n,y,b,r),h.push(I)),I}function T(b){if(--b.usedTimes===0){let y=h.indexOf(b);h[y]=h[h.length-1],h.pop(),b.destroy()}}function R(b){c.remove(b)}function C(){c.dispose()}return{getParameters:m,getProgramCacheKey:p,getUniforms:x,acquireProgram:D,releaseProgram:T,releaseShaderCache:R,programs:h,dispose:C}}function a0(){let n=new WeakMap;function t(o){return n.has(o)}function e(o){let a=n.get(o);return a===void 0&&(a={},n.set(o,a)),a}function i(o){n.delete(o)}function s(o,a,c){n.get(o)[a]=c}function r(){n=new WeakMap}return{has:t,get:e,remove:i,update:s,dispose:r}}function c0(n,t){return n.groupOrder!==t.groupOrder?n.groupOrder-t.groupOrder:n.renderOrder!==t.renderOrder?n.renderOrder-t.renderOrder:n.material.id!==t.material.id?n.material.id-t.material.id:n.z!==t.z?n.z-t.z:n.id-t.id}function sh(n,t){return n.groupOrder!==t.groupOrder?n.groupOrder-t.groupOrder:n.renderOrder!==t.renderOrder?n.renderOrder-t.renderOrder:n.z!==t.z?t.z-n.z:n.id-t.id}function rh(){let n=[],t=0,e=[],i=[],s=[];function r(){t=0,e.length=0,i.length=0,s.length=0}function o(u,d,f,g,_,m){let p=n[t];return p===void 0?(p={id:u.id,object:u,geometry:d,material:f,groupOrder:g,renderOrder:u.renderOrder,z:_,group:m},n[t]=p):(p.id=u.id,p.object=u,p.geometry=d,p.material=f,p.groupOrder=g,p.renderOrder=u.renderOrder,p.z=_,p.group=m),t++,p}function a(u,d,f,g,_,m){let p=o(u,d,f,g,_,m);f.transmission>0?i.push(p):f.transparent===!0?s.push(p):e.push(p)}function c(u,d,f,g,_,m){let p=o(u,d,f,g,_,m);f.transmission>0?i.unshift(p):f.transparent===!0?s.unshift(p):e.unshift(p)}function l(u,d){e.length>1&&e.sort(u||c0),i.length>1&&i.sort(d||sh),s.length>1&&s.sort(d||sh)}function h(){for(let u=t,d=n.length;u<d;u++){let f=n[u];if(f.id===null)break;f.id=null,f.object=null,f.geometry=null,f.material=null,f.group=null}}return{opaque:e,transmissive:i,transparent:s,init:r,push:a,unshift:c,finish:h,sort:l}}function l0(){let n=new WeakMap;function t(i,s){let r=n.get(i),o;return r===void 0?(o=new rh,n.set(i,[o])):s>=r.length?(o=new rh,r.push(o)):o=r[s],o}function e(){n=new WeakMap}return{get:t,dispose:e}}function h0(){let n={};return{get:function(t){if(n[t.id]!==void 0)return n[t.id];let e;switch(t.type){case"DirectionalLight":e={direction:new L,color:new kt};break;case"SpotLight":e={position:new L,direction:new L,color:new kt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":e={position:new L,color:new kt,distance:0,decay:0};break;case"HemisphereLight":e={direction:new L,skyColor:new kt,groundColor:new kt};break;case"RectAreaLight":e={color:new kt,position:new L,halfWidth:new L,halfHeight:new L};break}return n[t.id]=e,e}}}function u0(){let n={};return{get:function(t){if(n[t.id]!==void 0)return n[t.id];let e;switch(t.type){case"DirectionalLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new j};break;case"SpotLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new j};break;case"PointLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new j,shadowCameraNear:1,shadowCameraFar:1e3};break}return n[t.id]=e,e}}}var d0=0;function f0(n,t){return(t.castShadow?2:0)-(n.castShadow?2:0)+(t.map?1:0)-(n.map?1:0)}function p0(n){let t=new h0,e=u0(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)i.probe.push(new L);let s=new L,r=new de,o=new de;function a(l){let h=0,u=0,d=0;for(let b=0;b<9;b++)i.probe[b].set(0,0,0);let f=0,g=0,_=0,m=0,p=0,S=0,M=0,x=0,D=0,T=0,R=0;l.sort(f0);for(let b=0,y=l.length;b<y;b++){let I=l[b],B=I.color,N=I.intensity,H=I.distance,Y=I.shadow&&I.shadow.map?I.shadow.map.texture:null;if(I.isAmbientLight)h+=B.r*N,u+=B.g*N,d+=B.b*N;else if(I.isLightProbe){for(let V=0;V<9;V++)i.probe[V].addScaledVector(I.sh.coefficients[V],N);R++}else if(I.isDirectionalLight){let V=t.get(I);if(V.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){let rt=I.shadow,X=e.get(I);X.shadowIntensity=rt.intensity,X.shadowBias=rt.bias,X.shadowNormalBias=rt.normalBias,X.shadowRadius=rt.radius,X.shadowMapSize=rt.mapSize,i.directionalShadow[f]=X,i.directionalShadowMap[f]=Y,i.directionalShadowMatrix[f]=I.shadow.matrix,S++}i.directional[f]=V,f++}else if(I.isSpotLight){let V=t.get(I);V.position.setFromMatrixPosition(I.matrixWorld),V.color.copy(B).multiplyScalar(N),V.distance=H,V.coneCos=Math.cos(I.angle),V.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),V.decay=I.decay,i.spot[_]=V;let rt=I.shadow;if(I.map&&(i.spotLightMap[D]=I.map,D++,rt.updateMatrices(I),I.castShadow&&T++),i.spotLightMatrix[_]=rt.matrix,I.castShadow){let X=e.get(I);X.shadowIntensity=rt.intensity,X.shadowBias=rt.bias,X.shadowNormalBias=rt.normalBias,X.shadowRadius=rt.radius,X.shadowMapSize=rt.mapSize,i.spotShadow[_]=X,i.spotShadowMap[_]=Y,x++}_++}else if(I.isRectAreaLight){let V=t.get(I);V.color.copy(B).multiplyScalar(N),V.halfWidth.set(I.width*.5,0,0),V.halfHeight.set(0,I.height*.5,0),i.rectArea[m]=V,m++}else if(I.isPointLight){let V=t.get(I);if(V.color.copy(I.color).multiplyScalar(I.intensity),V.distance=I.distance,V.decay=I.decay,I.castShadow){let rt=I.shadow,X=e.get(I);X.shadowIntensity=rt.intensity,X.shadowBias=rt.bias,X.shadowNormalBias=rt.normalBias,X.shadowRadius=rt.radius,X.shadowMapSize=rt.mapSize,X.shadowCameraNear=rt.camera.near,X.shadowCameraFar=rt.camera.far,i.pointShadow[g]=X,i.pointShadowMap[g]=Y,i.pointShadowMatrix[g]=I.shadow.matrix,M++}i.point[g]=V,g++}else if(I.isHemisphereLight){let V=t.get(I);V.skyColor.copy(I.color).multiplyScalar(N),V.groundColor.copy(I.groundColor).multiplyScalar(N),i.hemi[p]=V,p++}}m>0&&(n.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=ut.LTC_FLOAT_1,i.rectAreaLTC2=ut.LTC_FLOAT_2):(i.rectAreaLTC1=ut.LTC_HALF_1,i.rectAreaLTC2=ut.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=u,i.ambient[2]=d;let C=i.hash;(C.directionalLength!==f||C.pointLength!==g||C.spotLength!==_||C.rectAreaLength!==m||C.hemiLength!==p||C.numDirectionalShadows!==S||C.numPointShadows!==M||C.numSpotShadows!==x||C.numSpotMaps!==D||C.numLightProbes!==R)&&(i.directional.length=f,i.spot.length=_,i.rectArea.length=m,i.point.length=g,i.hemi.length=p,i.directionalShadow.length=S,i.directionalShadowMap.length=S,i.pointShadow.length=M,i.pointShadowMap.length=M,i.spotShadow.length=x,i.spotShadowMap.length=x,i.directionalShadowMatrix.length=S,i.pointShadowMatrix.length=M,i.spotLightMatrix.length=x+D-T,i.spotLightMap.length=D,i.numSpotLightShadowsWithMaps=T,i.numLightProbes=R,C.directionalLength=f,C.pointLength=g,C.spotLength=_,C.rectAreaLength=m,C.hemiLength=p,C.numDirectionalShadows=S,C.numPointShadows=M,C.numSpotShadows=x,C.numSpotMaps=D,C.numLightProbes=R,i.version=d0++)}function c(l,h){let u=0,d=0,f=0,g=0,_=0,m=h.matrixWorldInverse;for(let p=0,S=l.length;p<S;p++){let M=l[p];if(M.isDirectionalLight){let x=i.directional[u];x.direction.setFromMatrixPosition(M.matrixWorld),s.setFromMatrixPosition(M.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),u++}else if(M.isSpotLight){let x=i.spot[f];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),x.direction.setFromMatrixPosition(M.matrixWorld),s.setFromMatrixPosition(M.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),f++}else if(M.isRectAreaLight){let x=i.rectArea[g];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),o.identity(),r.copy(M.matrixWorld),r.premultiply(m),o.extractRotation(r),x.halfWidth.set(M.width*.5,0,0),x.halfHeight.set(0,M.height*.5,0),x.halfWidth.applyMatrix4(o),x.halfHeight.applyMatrix4(o),g++}else if(M.isPointLight){let x=i.point[d];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),d++}else if(M.isHemisphereLight){let x=i.hemi[_];x.direction.setFromMatrixPosition(M.matrixWorld),x.direction.transformDirection(m),_++}}}return{setup:a,setupView:c,state:i}}function oh(n){let t=new p0(n),e=[],i=[];function s(h){l.camera=h,e.length=0,i.length=0}function r(h){e.push(h)}function o(h){i.push(h)}function a(){t.setup(e)}function c(h){t.setupView(e,h)}let l={lightsArray:e,shadowsArray:i,camera:null,lights:t,transmissionRenderTarget:{}};return{init:s,state:l,setupLights:a,setupLightsView:c,pushLight:r,pushShadow:o}}function m0(n){let t=new WeakMap;function e(s,r=0){let o=t.get(s),a;return o===void 0?(a=new oh(n),t.set(s,[a])):r>=o.length?(a=new oh(n),o.push(a)):a=o[r],a}function i(){t=new WeakMap}return{get:e,dispose:i}}var Za=class extends Si{static get type(){return"MeshDepthMaterial"}constructor(t){super(),this.isMeshDepthMaterial=!0,this.depthPacking=od,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}},$a=class extends Si{static get type(){return"MeshDistanceMaterial"}constructor(t){super(),this.isMeshDistanceMaterial=!0,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}},g0=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,_0=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function x0(n,t,e){let i=new Bs,s=new j,r=new j,o=new ie,a=new Za({depthPacking:ad}),c=new $a,l={},h=e.maxTextureSize,u={[$n]:Be,[Be]:$n,[ve]:ve},d=new dn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new j},radius:{value:4}},vertexShader:g0,fragmentShader:_0}),f=d.clone();f.defines.HORIZONTAL_PASS=1;let g=new Te;g.setAttribute("position",new we(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let _=new Pt(g,d),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=xh;let p=this.type;this.render=function(T,R,C){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||T.length===0)return;let b=n.getRenderTarget(),y=n.getActiveCubeFace(),I=n.getActiveMipmapLevel(),B=n.state;B.setBlending(qn),B.buffers.color.setClear(1,1,1,1),B.buffers.depth.setTest(!0),B.setScissorTest(!1);let N=p!==Tn&&this.type===Tn,H=p===Tn&&this.type!==Tn;for(let Y=0,V=T.length;Y<V;Y++){let rt=T[Y],X=rt.shadow;if(X===void 0){console.warn("THREE.WebGLShadowMap:",rt,"has no shadow.");continue}if(X.autoUpdate===!1&&X.needsUpdate===!1)continue;s.copy(X.mapSize);let it=X.getFrameExtents();if(s.multiply(it),r.copy(X.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/it.x),s.x=r.x*it.x,X.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/it.y),s.y=r.y*it.y,X.mapSize.y=r.y)),X.map===null||N===!0||H===!0){let vt=this.type!==Tn?{minFilter:rn,magFilter:rn}:{};X.map!==null&&X.map.dispose(),X.map=new Ln(s.x,s.y,vt),X.map.texture.name=rt.name+".shadowMap",X.camera.updateProjectionMatrix()}n.setRenderTarget(X.map),n.clear();let dt=X.getViewportCount();for(let vt=0;vt<dt;vt++){let zt=X.getViewport(vt);o.set(r.x*zt.x,r.y*zt.y,r.x*zt.z,r.y*zt.w),B.viewport(o),X.updateMatrices(rt,vt),i=X.getFrustum(),x(R,C,X.camera,rt,this.type)}X.isPointLightShadow!==!0&&this.type===Tn&&S(X,C),X.needsUpdate=!1}p=this.type,m.needsUpdate=!1,n.setRenderTarget(b,y,I)};function S(T,R){let C=t.update(_);d.defines.VSM_SAMPLES!==T.blurSamples&&(d.defines.VSM_SAMPLES=T.blurSamples,f.defines.VSM_SAMPLES=T.blurSamples,d.needsUpdate=!0,f.needsUpdate=!0),T.mapPass===null&&(T.mapPass=new Ln(s.x,s.y)),d.uniforms.shadow_pass.value=T.map.texture,d.uniforms.resolution.value=T.mapSize,d.uniforms.radius.value=T.radius,n.setRenderTarget(T.mapPass),n.clear(),n.renderBufferDirect(R,null,C,d,_,null),f.uniforms.shadow_pass.value=T.mapPass.texture,f.uniforms.resolution.value=T.mapSize,f.uniforms.radius.value=T.radius,n.setRenderTarget(T.map),n.clear(),n.renderBufferDirect(R,null,C,f,_,null)}function M(T,R,C,b){let y=null,I=C.isPointLight===!0?T.customDistanceMaterial:T.customDepthMaterial;if(I!==void 0)y=I;else if(y=C.isPointLight===!0?c:a,n.localClippingEnabled&&R.clipShadows===!0&&Array.isArray(R.clippingPlanes)&&R.clippingPlanes.length!==0||R.displacementMap&&R.displacementScale!==0||R.alphaMap&&R.alphaTest>0||R.map&&R.alphaTest>0){let B=y.uuid,N=R.uuid,H=l[B];H===void 0&&(H={},l[B]=H);let Y=H[N];Y===void 0&&(Y=y.clone(),H[N]=Y,R.addEventListener("dispose",D)),y=Y}if(y.visible=R.visible,y.wireframe=R.wireframe,b===Tn?y.side=R.shadowSide!==null?R.shadowSide:R.side:y.side=R.shadowSide!==null?R.shadowSide:u[R.side],y.alphaMap=R.alphaMap,y.alphaTest=R.alphaTest,y.map=R.map,y.clipShadows=R.clipShadows,y.clippingPlanes=R.clippingPlanes,y.clipIntersection=R.clipIntersection,y.displacementMap=R.displacementMap,y.displacementScale=R.displacementScale,y.displacementBias=R.displacementBias,y.wireframeLinewidth=R.wireframeLinewidth,y.linewidth=R.linewidth,C.isPointLight===!0&&y.isMeshDistanceMaterial===!0){let B=n.properties.get(y);B.light=C}return y}function x(T,R,C,b,y){if(T.visible===!1)return;if(T.layers.test(R.layers)&&(T.isMesh||T.isLine||T.isPoints)&&(T.castShadow||T.receiveShadow&&y===Tn)&&(!T.frustumCulled||i.intersectsObject(T))){T.modelViewMatrix.multiplyMatrices(C.matrixWorldInverse,T.matrixWorld);let N=t.update(T),H=T.material;if(Array.isArray(H)){let Y=N.groups;for(let V=0,rt=Y.length;V<rt;V++){let X=Y[V],it=H[X.materialIndex];if(it&&it.visible){let dt=M(T,it,b,y);T.onBeforeShadow(n,T,R,C,N,dt,X),n.renderBufferDirect(C,null,N,dt,T,X),T.onAfterShadow(n,T,R,C,N,dt,X)}}}else if(H.visible){let Y=M(T,H,b,y);T.onBeforeShadow(n,T,R,C,N,Y,null),n.renderBufferDirect(C,null,N,Y,T,null),T.onAfterShadow(n,T,R,C,N,Y,null)}}let B=T.children;for(let N=0,H=B.length;N<H;N++)x(B[N],R,C,b,y)}function D(T){T.target.removeEventListener("dispose",D);for(let C in l){let b=l[C],y=T.target.uuid;y in b&&(b[y].dispose(),delete b[y])}}}var y0={[ta]:ea,[na]:ra,[ia]:oa,[Qi]:sa,[ea]:ta,[ra]:na,[oa]:ia,[sa]:Qi};function v0(n,t){function e(){let U=!1,pt=new ie,q=null,Q=new ie(0,0,0,0);return{setMask:function(_t){q!==_t&&!U&&(n.colorMask(_t,_t,_t,_t),q=_t)},setLocked:function(_t){U=_t},setClear:function(_t,mt,Vt,me,Re){Re===!0&&(_t*=me,mt*=me,Vt*=me),pt.set(_t,mt,Vt,me),Q.equals(pt)===!1&&(n.clearColor(_t,mt,Vt,me),Q.copy(pt))},reset:function(){U=!1,q=null,Q.set(-1,0,0,0)}}}function i(){let U=!1,pt=!1,q=null,Q=null,_t=null;return{setReversed:function(mt){if(pt!==mt){let Vt=t.get("EXT_clip_control");pt?Vt.clipControlEXT(Vt.LOWER_LEFT_EXT,Vt.ZERO_TO_ONE_EXT):Vt.clipControlEXT(Vt.LOWER_LEFT_EXT,Vt.NEGATIVE_ONE_TO_ONE_EXT);let me=_t;_t=null,this.setClear(me)}pt=mt},getReversed:function(){return pt},setTest:function(mt){mt?tt(n.DEPTH_TEST):wt(n.DEPTH_TEST)},setMask:function(mt){q!==mt&&!U&&(n.depthMask(mt),q=mt)},setFunc:function(mt){if(pt&&(mt=y0[mt]),Q!==mt){switch(mt){case ta:n.depthFunc(n.NEVER);break;case ea:n.depthFunc(n.ALWAYS);break;case na:n.depthFunc(n.LESS);break;case Qi:n.depthFunc(n.LEQUAL);break;case ia:n.depthFunc(n.EQUAL);break;case sa:n.depthFunc(n.GEQUAL);break;case ra:n.depthFunc(n.GREATER);break;case oa:n.depthFunc(n.NOTEQUAL);break;default:n.depthFunc(n.LEQUAL)}Q=mt}},setLocked:function(mt){U=mt},setClear:function(mt){_t!==mt&&(pt&&(mt=1-mt),n.clearDepth(mt),_t=mt)},reset:function(){U=!1,q=null,Q=null,_t=null,pt=!1}}}function s(){let U=!1,pt=null,q=null,Q=null,_t=null,mt=null,Vt=null,me=null,Re=null;return{setTest:function(ee){U||(ee?tt(n.STENCIL_TEST):wt(n.STENCIL_TEST))},setMask:function(ee){pt!==ee&&!U&&(n.stencilMask(ee),pt=ee)},setFunc:function(ee,Qe,yn){(q!==ee||Q!==Qe||_t!==yn)&&(n.stencilFunc(ee,Qe,yn),q=ee,Q=Qe,_t=yn)},setOp:function(ee,Qe,yn){(mt!==ee||Vt!==Qe||me!==yn)&&(n.stencilOp(ee,Qe,yn),mt=ee,Vt=Qe,me=yn)},setLocked:function(ee){U=ee},setClear:function(ee){Re!==ee&&(n.clearStencil(ee),Re=ee)},reset:function(){U=!1,pt=null,q=null,Q=null,_t=null,mt=null,Vt=null,me=null,Re=null}}}let r=new e,o=new i,a=new s,c=new WeakMap,l=new WeakMap,h={},u={},d=new WeakMap,f=[],g=null,_=!1,m=null,p=null,S=null,M=null,x=null,D=null,T=null,R=new kt(0,0,0),C=0,b=!1,y=null,I=null,B=null,N=null,H=null,Y=n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS),V=!1,rt=0,X=n.getParameter(n.VERSION);X.indexOf("WebGL")!==-1?(rt=parseFloat(/^WebGL (\d)/.exec(X)[1]),V=rt>=1):X.indexOf("OpenGL ES")!==-1&&(rt=parseFloat(/^OpenGL ES (\d)/.exec(X)[1]),V=rt>=2);let it=null,dt={},vt=n.getParameter(n.SCISSOR_BOX),zt=n.getParameter(n.VIEWPORT),Kt=new ie().fromArray(vt),P=new ie().fromArray(zt);function k(U,pt,q,Q){let _t=new Uint8Array(4),mt=n.createTexture();n.bindTexture(U,mt),n.texParameteri(U,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(U,n.TEXTURE_MAG_FILTER,n.NEAREST);for(let Vt=0;Vt<q;Vt++)U===n.TEXTURE_3D||U===n.TEXTURE_2D_ARRAY?n.texImage3D(pt,0,n.RGBA,1,1,Q,0,n.RGBA,n.UNSIGNED_BYTE,_t):n.texImage2D(pt+Vt,0,n.RGBA,1,1,0,n.RGBA,n.UNSIGNED_BYTE,_t);return mt}let nt={};nt[n.TEXTURE_2D]=k(n.TEXTURE_2D,n.TEXTURE_2D,1),nt[n.TEXTURE_CUBE_MAP]=k(n.TEXTURE_CUBE_MAP,n.TEXTURE_CUBE_MAP_POSITIVE_X,6),nt[n.TEXTURE_2D_ARRAY]=k(n.TEXTURE_2D_ARRAY,n.TEXTURE_2D_ARRAY,1,1),nt[n.TEXTURE_3D]=k(n.TEXTURE_3D,n.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),tt(n.DEPTH_TEST),o.setFunc(Qi),ot(!1),Mt(al),tt(n.CULL_FACE),A(qn);function tt(U){h[U]!==!0&&(n.enable(U),h[U]=!0)}function wt(U){h[U]!==!1&&(n.disable(U),h[U]=!1)}function Dt(U,pt){return u[U]!==pt?(n.bindFramebuffer(U,pt),u[U]=pt,U===n.DRAW_FRAMEBUFFER&&(u[n.FRAMEBUFFER]=pt),U===n.FRAMEBUFFER&&(u[n.DRAW_FRAMEBUFFER]=pt),!0):!1}function Lt(U,pt){let q=f,Q=!1;if(U){q=d.get(pt),q===void 0&&(q=[],d.set(pt,q));let _t=U.textures;if(q.length!==_t.length||q[0]!==n.COLOR_ATTACHMENT0){for(let mt=0,Vt=_t.length;mt<Vt;mt++)q[mt]=n.COLOR_ATTACHMENT0+mt;q.length=_t.length,Q=!0}}else q[0]!==n.BACK&&(q[0]=n.BACK,Q=!0);Q&&n.drawBuffers(q)}function Yt(U){return g!==U?(n.useProgram(U),g=U,!0):!1}let Z={[gi]:n.FUNC_ADD,[Uu]:n.FUNC_SUBTRACT,[Nu]:n.FUNC_REVERSE_SUBTRACT};Z[Fu]=n.MIN,Z[Ou]=n.MAX;let st={[Bu]:n.ZERO,[zu]:n.ONE,[Hu]:n.SRC_COLOR,[Qo]:n.SRC_ALPHA,[qu]:n.SRC_ALPHA_SATURATE,[Wu]:n.DST_COLOR,[Vu]:n.DST_ALPHA,[ku]:n.ONE_MINUS_SRC_COLOR,[jo]:n.ONE_MINUS_SRC_ALPHA,[Xu]:n.ONE_MINUS_DST_COLOR,[Gu]:n.ONE_MINUS_DST_ALPHA,[Yu]:n.CONSTANT_COLOR,[Zu]:n.ONE_MINUS_CONSTANT_COLOR,[$u]:n.CONSTANT_ALPHA,[Ju]:n.ONE_MINUS_CONSTANT_ALPHA};function A(U,pt,q,Q,_t,mt,Vt,me,Re,ee){if(U===qn){_===!0&&(wt(n.BLEND),_=!1);return}if(_===!1&&(tt(n.BLEND),_=!0),U!==Du){if(U!==m||ee!==b){if((p!==gi||x!==gi)&&(n.blendEquation(n.FUNC_ADD),p=gi,x=gi),ee)switch(U){case Zi:n.blendFuncSeparate(n.ONE,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case cl:n.blendFunc(n.ONE,n.ONE);break;case ll:n.blendFuncSeparate(n.ZERO,n.ONE_MINUS_SRC_COLOR,n.ZERO,n.ONE);break;case hl:n.blendFuncSeparate(n.ZERO,n.SRC_COLOR,n.ZERO,n.SRC_ALPHA);break;default:console.error("THREE.WebGLState: Invalid blending: ",U);break}else switch(U){case Zi:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case cl:n.blendFunc(n.SRC_ALPHA,n.ONE);break;case ll:n.blendFuncSeparate(n.ZERO,n.ONE_MINUS_SRC_COLOR,n.ZERO,n.ONE);break;case hl:n.blendFunc(n.ZERO,n.SRC_COLOR);break;default:console.error("THREE.WebGLState: Invalid blending: ",U);break}S=null,M=null,D=null,T=null,R.set(0,0,0),C=0,m=U,b=ee}return}_t=_t||pt,mt=mt||q,Vt=Vt||Q,(pt!==p||_t!==x)&&(n.blendEquationSeparate(Z[pt],Z[_t]),p=pt,x=_t),(q!==S||Q!==M||mt!==D||Vt!==T)&&(n.blendFuncSeparate(st[q],st[Q],st[mt],st[Vt]),S=q,M=Q,D=mt,T=Vt),(me.equals(R)===!1||Re!==C)&&(n.blendColor(me.r,me.g,me.b,Re),R.copy(me),C=Re),m=U,b=!1}function St(U,pt){U.side===ve?wt(n.CULL_FACE):tt(n.CULL_FACE);let q=U.side===Be;pt&&(q=!q),ot(q),U.blending===Zi&&U.transparent===!1?A(qn):A(U.blending,U.blendEquation,U.blendSrc,U.blendDst,U.blendEquationAlpha,U.blendSrcAlpha,U.blendDstAlpha,U.blendColor,U.blendAlpha,U.premultipliedAlpha),o.setFunc(U.depthFunc),o.setTest(U.depthTest),o.setMask(U.depthWrite),r.setMask(U.colorWrite);let Q=U.stencilWrite;a.setTest(Q),Q&&(a.setMask(U.stencilWriteMask),a.setFunc(U.stencilFunc,U.stencilRef,U.stencilFuncMask),a.setOp(U.stencilFail,U.stencilZFail,U.stencilZPass)),Ut(U.polygonOffset,U.polygonOffsetFactor,U.polygonOffsetUnits),U.alphaToCoverage===!0?tt(n.SAMPLE_ALPHA_TO_COVERAGE):wt(n.SAMPLE_ALPHA_TO_COVERAGE)}function ot(U){y!==U&&(U?n.frontFace(n.CW):n.frontFace(n.CCW),y=U)}function Mt(U){U!==Pu?(tt(n.CULL_FACE),U!==I&&(U===al?n.cullFace(n.BACK):U===Lu?n.cullFace(n.FRONT):n.cullFace(n.FRONT_AND_BACK))):wt(n.CULL_FACE),I=U}function ht(U){U!==B&&(V&&n.lineWidth(U),B=U)}function Ut(U,pt,q){U?(tt(n.POLYGON_OFFSET_FILL),(N!==pt||H!==q)&&(n.polygonOffset(pt,q),N=pt,H=q)):wt(n.POLYGON_OFFSET_FILL)}function xt(U){U?tt(n.SCISSOR_TEST):wt(n.SCISSOR_TEST)}function w(U){U===void 0&&(U=n.TEXTURE0+Y-1),it!==U&&(n.activeTexture(U),it=U)}function v(U,pt,q){q===void 0&&(it===null?q=n.TEXTURE0+Y-1:q=it);let Q=dt[q];Q===void 0&&(Q={type:void 0,texture:void 0},dt[q]=Q),(Q.type!==U||Q.texture!==pt)&&(it!==q&&(n.activeTexture(q),it=q),n.bindTexture(U,pt||nt[U]),Q.type=U,Q.texture=pt)}function z(){let U=dt[it];U!==void 0&&U.type!==void 0&&(n.bindTexture(U.type,null),U.type=void 0,U.texture=void 0)}function $(){try{n.compressedTexImage2D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function et(){try{n.compressedTexImage3D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function J(){try{n.texSubImage2D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Tt(){try{n.texSubImage3D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function ft(){try{n.compressedTexSubImage2D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function yt(){try{n.compressedTexSubImage3D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Zt(){try{n.texStorage2D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function at(){try{n.texStorage3D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Et(){try{n.texImage2D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Nt(){try{n.texImage3D.apply(n,arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Bt(U){Kt.equals(U)===!1&&(n.scissor(U.x,U.y,U.z,U.w),Kt.copy(U))}function bt(U){P.equals(U)===!1&&(n.viewport(U.x,U.y,U.z,U.w),P.copy(U))}function $t(U,pt){let q=l.get(pt);q===void 0&&(q=new WeakMap,l.set(pt,q));let Q=q.get(U);Q===void 0&&(Q=n.getUniformBlockIndex(pt,U.name),q.set(U,Q))}function Xt(U,pt){let Q=l.get(pt).get(U);c.get(pt)!==Q&&(n.uniformBlockBinding(pt,Q,U.__bindingPointIndex),c.set(pt,Q))}function oe(){n.disable(n.BLEND),n.disable(n.CULL_FACE),n.disable(n.DEPTH_TEST),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SCISSOR_TEST),n.disable(n.STENCIL_TEST),n.disable(n.SAMPLE_ALPHA_TO_COVERAGE),n.blendEquation(n.FUNC_ADD),n.blendFunc(n.ONE,n.ZERO),n.blendFuncSeparate(n.ONE,n.ZERO,n.ONE,n.ZERO),n.blendColor(0,0,0,0),n.colorMask(!0,!0,!0,!0),n.clearColor(0,0,0,0),n.depthMask(!0),n.depthFunc(n.LESS),o.setReversed(!1),n.clearDepth(1),n.stencilMask(4294967295),n.stencilFunc(n.ALWAYS,0,4294967295),n.stencilOp(n.KEEP,n.KEEP,n.KEEP),n.clearStencil(0),n.cullFace(n.BACK),n.frontFace(n.CCW),n.polygonOffset(0,0),n.activeTexture(n.TEXTURE0),n.bindFramebuffer(n.FRAMEBUFFER,null),n.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),n.bindFramebuffer(n.READ_FRAMEBUFFER,null),n.useProgram(null),n.lineWidth(1),n.scissor(0,0,n.canvas.width,n.canvas.height),n.viewport(0,0,n.canvas.width,n.canvas.height),h={},it=null,dt={},u={},d=new WeakMap,f=[],g=null,_=!1,m=null,p=null,S=null,M=null,x=null,D=null,T=null,R=new kt(0,0,0),C=0,b=!1,y=null,I=null,B=null,N=null,H=null,Kt.set(0,0,n.canvas.width,n.canvas.height),P.set(0,0,n.canvas.width,n.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:tt,disable:wt,bindFramebuffer:Dt,drawBuffers:Lt,useProgram:Yt,setBlending:A,setMaterial:St,setFlipSided:ot,setCullFace:Mt,setLineWidth:ht,setPolygonOffset:Ut,setScissorTest:xt,activeTexture:w,bindTexture:v,unbindTexture:z,compressedTexImage2D:$,compressedTexImage3D:et,texImage2D:Et,texImage3D:Nt,updateUBOMapping:$t,uniformBlockBinding:Xt,texStorage2D:Zt,texStorage3D:at,texSubImage2D:J,texSubImage3D:Tt,compressedTexSubImage2D:ft,compressedTexSubImage3D:yt,scissor:Bt,viewport:bt,reset:oe}}function ah(n,t,e,i){let s=M0(i);switch(e){case bh:return n*t;case Th:return n*t;case Ah:return n*t*2;case Rh:return n*t/s.components*s.byteLength;case Cc:return n*t/s.components*s.byteLength;case Ch:return n*t*2/s.components*s.byteLength;case Ic:return n*t*2/s.components*s.byteLength;case wh:return n*t*3/s.components*s.byteLength;case sn:return n*t*4/s.components*s.byteLength;case Pc:return n*t*4/s.components*s.byteLength;case Sr:case Er:return Math.floor((n+3)/4)*Math.floor((t+3)/4)*8;case br:case wr:return Math.floor((n+3)/4)*Math.floor((t+3)/4)*16;case ua:case fa:return Math.max(n,16)*Math.max(t,8)/4;case ha:case da:return Math.max(n,8)*Math.max(t,8)/2;case pa:case ma:return Math.floor((n+3)/4)*Math.floor((t+3)/4)*8;case ga:return Math.floor((n+3)/4)*Math.floor((t+3)/4)*16;case _a:return Math.floor((n+3)/4)*Math.floor((t+3)/4)*16;case xa:return Math.floor((n+4)/5)*Math.floor((t+3)/4)*16;case ya:return Math.floor((n+4)/5)*Math.floor((t+4)/5)*16;case va:return Math.floor((n+5)/6)*Math.floor((t+4)/5)*16;case Ma:return Math.floor((n+5)/6)*Math.floor((t+5)/6)*16;case Sa:return Math.floor((n+7)/8)*Math.floor((t+4)/5)*16;case Ea:return Math.floor((n+7)/8)*Math.floor((t+5)/6)*16;case ba:return Math.floor((n+7)/8)*Math.floor((t+7)/8)*16;case wa:return Math.floor((n+9)/10)*Math.floor((t+4)/5)*16;case Ta:return Math.floor((n+9)/10)*Math.floor((t+5)/6)*16;case Aa:return Math.floor((n+9)/10)*Math.floor((t+7)/8)*16;case Ra:return Math.floor((n+9)/10)*Math.floor((t+9)/10)*16;case Ca:return Math.floor((n+11)/12)*Math.floor((t+9)/10)*16;case Ia:return Math.floor((n+11)/12)*Math.floor((t+11)/12)*16;case Tr:case Pa:case La:return Math.ceil(n/4)*Math.ceil(t/4)*16;case Ih:case Da:return Math.ceil(n/4)*Math.ceil(t/4)*8;case Ua:case Na:return Math.ceil(n/4)*Math.ceil(t/4)*16}throw new Error(`Unable to determine texture byte length for ${e} format.`)}function M0(n){switch(n){case Pn:case Mh:return{byteLength:1,components:1};case Us:case Sh:case Gs:return{byteLength:2,components:1};case Ac:case Rc:return{byteLength:2,components:4};case Mi:case Tc:case Rn:return{byteLength:4,components:1};case Eh:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${n}.`)}function S0(n,t,e,i,s,r,o){let a=t.has("WEBGL_multisampled_render_to_texture")?t.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new j,h=new WeakMap,u,d=new WeakMap,f=!1;try{f=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function g(w,v){return f?new OffscreenCanvas(w,v):Cr("canvas")}function _(w,v,z){let $=1,et=xt(w);if((et.width>z||et.height>z)&&($=z/Math.max(et.width,et.height)),$<1)if(typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&w instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&w instanceof ImageBitmap||typeof VideoFrame<"u"&&w instanceof VideoFrame){let J=Math.floor($*et.width),Tt=Math.floor($*et.height);u===void 0&&(u=g(J,Tt));let ft=v?g(J,Tt):u;return ft.width=J,ft.height=Tt,ft.getContext("2d").drawImage(w,0,0,J,Tt),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+et.width+"x"+et.height+") to ("+J+"x"+Tt+")."),ft}else return"data"in w&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+et.width+"x"+et.height+")."),w;return w}function m(w){return w.generateMipmaps}function p(w){n.generateMipmap(w)}function S(w){return w.isWebGLCubeRenderTarget?n.TEXTURE_CUBE_MAP:w.isWebGL3DRenderTarget?n.TEXTURE_3D:w.isWebGLArrayRenderTarget||w.isCompressedArrayTexture?n.TEXTURE_2D_ARRAY:n.TEXTURE_2D}function M(w,v,z,$,et=!1){if(w!==null){if(n[w]!==void 0)return n[w];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+w+"'")}let J=v;if(v===n.RED&&(z===n.FLOAT&&(J=n.R32F),z===n.HALF_FLOAT&&(J=n.R16F),z===n.UNSIGNED_BYTE&&(J=n.R8)),v===n.RED_INTEGER&&(z===n.UNSIGNED_BYTE&&(J=n.R8UI),z===n.UNSIGNED_SHORT&&(J=n.R16UI),z===n.UNSIGNED_INT&&(J=n.R32UI),z===n.BYTE&&(J=n.R8I),z===n.SHORT&&(J=n.R16I),z===n.INT&&(J=n.R32I)),v===n.RG&&(z===n.FLOAT&&(J=n.RG32F),z===n.HALF_FLOAT&&(J=n.RG16F),z===n.UNSIGNED_BYTE&&(J=n.RG8)),v===n.RG_INTEGER&&(z===n.UNSIGNED_BYTE&&(J=n.RG8UI),z===n.UNSIGNED_SHORT&&(J=n.RG16UI),z===n.UNSIGNED_INT&&(J=n.RG32UI),z===n.BYTE&&(J=n.RG8I),z===n.SHORT&&(J=n.RG16I),z===n.INT&&(J=n.RG32I)),v===n.RGB_INTEGER&&(z===n.UNSIGNED_BYTE&&(J=n.RGB8UI),z===n.UNSIGNED_SHORT&&(J=n.RGB16UI),z===n.UNSIGNED_INT&&(J=n.RGB32UI),z===n.BYTE&&(J=n.RGB8I),z===n.SHORT&&(J=n.RGB16I),z===n.INT&&(J=n.RGB32I)),v===n.RGBA_INTEGER&&(z===n.UNSIGNED_BYTE&&(J=n.RGBA8UI),z===n.UNSIGNED_SHORT&&(J=n.RGBA16UI),z===n.UNSIGNED_INT&&(J=n.RGBA32UI),z===n.BYTE&&(J=n.RGBA8I),z===n.SHORT&&(J=n.RGBA16I),z===n.INT&&(J=n.RGBA32I)),v===n.RGB&&z===n.UNSIGNED_INT_5_9_9_9_REV&&(J=n.RGB9_E5),v===n.RGBA){let Tt=et?so:Jt.getTransfer($);z===n.FLOAT&&(J=n.RGBA32F),z===n.HALF_FLOAT&&(J=n.RGBA16F),z===n.UNSIGNED_BYTE&&(J=Tt===ne?n.SRGB8_ALPHA8:n.RGBA8),z===n.UNSIGNED_SHORT_4_4_4_4&&(J=n.RGBA4),z===n.UNSIGNED_SHORT_5_5_5_1&&(J=n.RGB5_A1)}return(J===n.R16F||J===n.R32F||J===n.RG16F||J===n.RG32F||J===n.RGBA16F||J===n.RGBA32F)&&t.get("EXT_color_buffer_float"),J}function x(w,v){let z;return w?v===null||v===Mi||v===es?z=n.DEPTH24_STENCIL8:v===Rn?z=n.DEPTH32F_STENCIL8:v===Us&&(z=n.DEPTH24_STENCIL8,console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===Mi||v===es?z=n.DEPTH_COMPONENT24:v===Rn?z=n.DEPTH_COMPONENT32F:v===Us&&(z=n.DEPTH_COMPONENT16),z}function D(w,v){return m(w)===!0||w.isFramebufferTexture&&w.minFilter!==rn&&w.minFilter!==hn?Math.log2(Math.max(v.width,v.height))+1:w.mipmaps!==void 0&&w.mipmaps.length>0?w.mipmaps.length:w.isCompressedTexture&&Array.isArray(w.image)?v.mipmaps.length:1}function T(w){let v=w.target;v.removeEventListener("dispose",T),C(v),v.isVideoTexture&&h.delete(v)}function R(w){let v=w.target;v.removeEventListener("dispose",R),y(v)}function C(w){let v=i.get(w);if(v.__webglInit===void 0)return;let z=w.source,$=d.get(z);if($){let et=$[v.__cacheKey];et.usedTimes--,et.usedTimes===0&&b(w),Object.keys($).length===0&&d.delete(z)}i.remove(w)}function b(w){let v=i.get(w);n.deleteTexture(v.__webglTexture);let z=w.source,$=d.get(z);delete $[v.__cacheKey],o.memory.textures--}function y(w){let v=i.get(w);if(w.depthTexture&&(w.depthTexture.dispose(),i.remove(w.depthTexture)),w.isWebGLCubeRenderTarget)for(let $=0;$<6;$++){if(Array.isArray(v.__webglFramebuffer[$]))for(let et=0;et<v.__webglFramebuffer[$].length;et++)n.deleteFramebuffer(v.__webglFramebuffer[$][et]);else n.deleteFramebuffer(v.__webglFramebuffer[$]);v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer[$])}else{if(Array.isArray(v.__webglFramebuffer))for(let $=0;$<v.__webglFramebuffer.length;$++)n.deleteFramebuffer(v.__webglFramebuffer[$]);else n.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&n.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let $=0;$<v.__webglColorRenderbuffer.length;$++)v.__webglColorRenderbuffer[$]&&n.deleteRenderbuffer(v.__webglColorRenderbuffer[$]);v.__webglDepthRenderbuffer&&n.deleteRenderbuffer(v.__webglDepthRenderbuffer)}let z=w.textures;for(let $=0,et=z.length;$<et;$++){let J=i.get(z[$]);J.__webglTexture&&(n.deleteTexture(J.__webglTexture),o.memory.textures--),i.remove(z[$])}i.remove(w)}let I=0;function B(){I=0}function N(){let w=I;return w>=s.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+w+" texture units while this GPU supports only "+s.maxTextures),I+=1,w}function H(w){let v=[];return v.push(w.wrapS),v.push(w.wrapT),v.push(w.wrapR||0),v.push(w.magFilter),v.push(w.minFilter),v.push(w.anisotropy),v.push(w.internalFormat),v.push(w.format),v.push(w.type),v.push(w.generateMipmaps),v.push(w.premultiplyAlpha),v.push(w.flipY),v.push(w.unpackAlignment),v.push(w.colorSpace),v.join()}function Y(w,v){let z=i.get(w);if(w.isVideoTexture&&ht(w),w.isRenderTargetTexture===!1&&w.version>0&&z.__version!==w.version){let $=w.image;if($===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if($.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{P(z,w,v);return}}e.bindTexture(n.TEXTURE_2D,z.__webglTexture,n.TEXTURE0+v)}function V(w,v){let z=i.get(w);if(w.version>0&&z.__version!==w.version){P(z,w,v);return}e.bindTexture(n.TEXTURE_2D_ARRAY,z.__webglTexture,n.TEXTURE0+v)}function rt(w,v){let z=i.get(w);if(w.version>0&&z.__version!==w.version){P(z,w,v);return}e.bindTexture(n.TEXTURE_3D,z.__webglTexture,n.TEXTURE0+v)}function X(w,v){let z=i.get(w);if(w.version>0&&z.__version!==w.version){k(z,w,v);return}e.bindTexture(n.TEXTURE_CUBE_MAP,z.__webglTexture,n.TEXTURE0+v)}let it={[Ds]:n.REPEAT,[yi]:n.CLAMP_TO_EDGE,[la]:n.MIRRORED_REPEAT},dt={[rn]:n.NEAREST,[rd]:n.NEAREST_MIPMAP_NEAREST,[Ks]:n.NEAREST_MIPMAP_LINEAR,[hn]:n.LINEAR,[yo]:n.LINEAR_MIPMAP_NEAREST,[vi]:n.LINEAR_MIPMAP_LINEAR},vt={[ld]:n.NEVER,[md]:n.ALWAYS,[hd]:n.LESS,[Lh]:n.LEQUAL,[ud]:n.EQUAL,[pd]:n.GEQUAL,[dd]:n.GREATER,[fd]:n.NOTEQUAL};function zt(w,v){if(v.type===Rn&&t.has("OES_texture_float_linear")===!1&&(v.magFilter===hn||v.magFilter===yo||v.magFilter===Ks||v.magFilter===vi||v.minFilter===hn||v.minFilter===yo||v.minFilter===Ks||v.minFilter===vi)&&console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),n.texParameteri(w,n.TEXTURE_WRAP_S,it[v.wrapS]),n.texParameteri(w,n.TEXTURE_WRAP_T,it[v.wrapT]),(w===n.TEXTURE_3D||w===n.TEXTURE_2D_ARRAY)&&n.texParameteri(w,n.TEXTURE_WRAP_R,it[v.wrapR]),n.texParameteri(w,n.TEXTURE_MAG_FILTER,dt[v.magFilter]),n.texParameteri(w,n.TEXTURE_MIN_FILTER,dt[v.minFilter]),v.compareFunction&&(n.texParameteri(w,n.TEXTURE_COMPARE_MODE,n.COMPARE_REF_TO_TEXTURE),n.texParameteri(w,n.TEXTURE_COMPARE_FUNC,vt[v.compareFunction])),t.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===rn||v.minFilter!==Ks&&v.minFilter!==vi||v.type===Rn&&t.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||i.get(v).__currentAnisotropy){let z=t.get("EXT_texture_filter_anisotropic");n.texParameterf(w,z.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,s.getMaxAnisotropy())),i.get(v).__currentAnisotropy=v.anisotropy}}}function Kt(w,v){let z=!1;w.__webglInit===void 0&&(w.__webglInit=!0,v.addEventListener("dispose",T));let $=v.source,et=d.get($);et===void 0&&(et={},d.set($,et));let J=H(v);if(J!==w.__cacheKey){et[J]===void 0&&(et[J]={texture:n.createTexture(),usedTimes:0},o.memory.textures++,z=!0),et[J].usedTimes++;let Tt=et[w.__cacheKey];Tt!==void 0&&(et[w.__cacheKey].usedTimes--,Tt.usedTimes===0&&b(v)),w.__cacheKey=J,w.__webglTexture=et[J].texture}return z}function P(w,v,z){let $=n.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&($=n.TEXTURE_2D_ARRAY),v.isData3DTexture&&($=n.TEXTURE_3D);let et=Kt(w,v),J=v.source;e.bindTexture($,w.__webglTexture,n.TEXTURE0+z);let Tt=i.get(J);if(J.version!==Tt.__version||et===!0){e.activeTexture(n.TEXTURE0+z);let ft=Jt.getPrimaries(Jt.workingColorSpace),yt=v.colorSpace===Xn?null:Jt.getPrimaries(v.colorSpace),Zt=v.colorSpace===Xn||ft===yt?n.NONE:n.BROWSER_DEFAULT_WEBGL;n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),n.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,Zt);let at=_(v.image,!1,s.maxTextureSize);at=Ut(v,at);let Et=r.convert(v.format,v.colorSpace),Nt=r.convert(v.type),Bt=M(v.internalFormat,Et,Nt,v.colorSpace,v.isVideoTexture);zt($,v);let bt,$t=v.mipmaps,Xt=v.isVideoTexture!==!0,oe=Tt.__version===void 0||et===!0,U=J.dataReady,pt=D(v,at);if(v.isDepthTexture)Bt=x(v.format===ns,v.type),oe&&(Xt?e.texStorage2D(n.TEXTURE_2D,1,Bt,at.width,at.height):e.texImage2D(n.TEXTURE_2D,0,Bt,at.width,at.height,0,Et,Nt,null));else if(v.isDataTexture)if($t.length>0){Xt&&oe&&e.texStorage2D(n.TEXTURE_2D,pt,Bt,$t[0].width,$t[0].height);for(let q=0,Q=$t.length;q<Q;q++)bt=$t[q],Xt?U&&e.texSubImage2D(n.TEXTURE_2D,q,0,0,bt.width,bt.height,Et,Nt,bt.data):e.texImage2D(n.TEXTURE_2D,q,Bt,bt.width,bt.height,0,Et,Nt,bt.data);v.generateMipmaps=!1}else Xt?(oe&&e.texStorage2D(n.TEXTURE_2D,pt,Bt,at.width,at.height),U&&e.texSubImage2D(n.TEXTURE_2D,0,0,0,at.width,at.height,Et,Nt,at.data)):e.texImage2D(n.TEXTURE_2D,0,Bt,at.width,at.height,0,Et,Nt,at.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){Xt&&oe&&e.texStorage3D(n.TEXTURE_2D_ARRAY,pt,Bt,$t[0].width,$t[0].height,at.depth);for(let q=0,Q=$t.length;q<Q;q++)if(bt=$t[q],v.format!==sn)if(Et!==null)if(Xt){if(U)if(v.layerUpdates.size>0){let _t=ah(bt.width,bt.height,v.format,v.type);for(let mt of v.layerUpdates){let Vt=bt.data.subarray(mt*_t/bt.data.BYTES_PER_ELEMENT,(mt+1)*_t/bt.data.BYTES_PER_ELEMENT);e.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,q,0,0,mt,bt.width,bt.height,1,Et,Vt)}v.clearLayerUpdates()}else e.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,q,0,0,0,bt.width,bt.height,at.depth,Et,bt.data)}else e.compressedTexImage3D(n.TEXTURE_2D_ARRAY,q,Bt,bt.width,bt.height,at.depth,0,bt.data,0,0);else console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else Xt?U&&e.texSubImage3D(n.TEXTURE_2D_ARRAY,q,0,0,0,bt.width,bt.height,at.depth,Et,Nt,bt.data):e.texImage3D(n.TEXTURE_2D_ARRAY,q,Bt,bt.width,bt.height,at.depth,0,Et,Nt,bt.data)}else{Xt&&oe&&e.texStorage2D(n.TEXTURE_2D,pt,Bt,$t[0].width,$t[0].height);for(let q=0,Q=$t.length;q<Q;q++)bt=$t[q],v.format!==sn?Et!==null?Xt?U&&e.compressedTexSubImage2D(n.TEXTURE_2D,q,0,0,bt.width,bt.height,Et,bt.data):e.compressedTexImage2D(n.TEXTURE_2D,q,Bt,bt.width,bt.height,0,bt.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Xt?U&&e.texSubImage2D(n.TEXTURE_2D,q,0,0,bt.width,bt.height,Et,Nt,bt.data):e.texImage2D(n.TEXTURE_2D,q,Bt,bt.width,bt.height,0,Et,Nt,bt.data)}else if(v.isDataArrayTexture)if(Xt){if(oe&&e.texStorage3D(n.TEXTURE_2D_ARRAY,pt,Bt,at.width,at.height,at.depth),U)if(v.layerUpdates.size>0){let q=ah(at.width,at.height,v.format,v.type);for(let Q of v.layerUpdates){let _t=at.data.subarray(Q*q/at.data.BYTES_PER_ELEMENT,(Q+1)*q/at.data.BYTES_PER_ELEMENT);e.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,Q,at.width,at.height,1,Et,Nt,_t)}v.clearLayerUpdates()}else e.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,0,at.width,at.height,at.depth,Et,Nt,at.data)}else e.texImage3D(n.TEXTURE_2D_ARRAY,0,Bt,at.width,at.height,at.depth,0,Et,Nt,at.data);else if(v.isData3DTexture)Xt?(oe&&e.texStorage3D(n.TEXTURE_3D,pt,Bt,at.width,at.height,at.depth),U&&e.texSubImage3D(n.TEXTURE_3D,0,0,0,0,at.width,at.height,at.depth,Et,Nt,at.data)):e.texImage3D(n.TEXTURE_3D,0,Bt,at.width,at.height,at.depth,0,Et,Nt,at.data);else if(v.isFramebufferTexture){if(oe)if(Xt)e.texStorage2D(n.TEXTURE_2D,pt,Bt,at.width,at.height);else{let q=at.width,Q=at.height;for(let _t=0;_t<pt;_t++)e.texImage2D(n.TEXTURE_2D,_t,Bt,q,Q,0,Et,Nt,null),q>>=1,Q>>=1}}else if($t.length>0){if(Xt&&oe){let q=xt($t[0]);e.texStorage2D(n.TEXTURE_2D,pt,Bt,q.width,q.height)}for(let q=0,Q=$t.length;q<Q;q++)bt=$t[q],Xt?U&&e.texSubImage2D(n.TEXTURE_2D,q,0,0,Et,Nt,bt):e.texImage2D(n.TEXTURE_2D,q,Bt,Et,Nt,bt);v.generateMipmaps=!1}else if(Xt){if(oe){let q=xt(at);e.texStorage2D(n.TEXTURE_2D,pt,Bt,q.width,q.height)}U&&e.texSubImage2D(n.TEXTURE_2D,0,0,0,Et,Nt,at)}else e.texImage2D(n.TEXTURE_2D,0,Bt,Et,Nt,at);m(v)&&p($),Tt.__version=J.version,v.onUpdate&&v.onUpdate(v)}w.__version=v.version}function k(w,v,z){if(v.image.length!==6)return;let $=Kt(w,v),et=v.source;e.bindTexture(n.TEXTURE_CUBE_MAP,w.__webglTexture,n.TEXTURE0+z);let J=i.get(et);if(et.version!==J.__version||$===!0){e.activeTexture(n.TEXTURE0+z);let Tt=Jt.getPrimaries(Jt.workingColorSpace),ft=v.colorSpace===Xn?null:Jt.getPrimaries(v.colorSpace),yt=v.colorSpace===Xn||Tt===ft?n.NONE:n.BROWSER_DEFAULT_WEBGL;n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),n.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,yt);let Zt=v.isCompressedTexture||v.image[0].isCompressedTexture,at=v.image[0]&&v.image[0].isDataTexture,Et=[];for(let Q=0;Q<6;Q++)!Zt&&!at?Et[Q]=_(v.image[Q],!0,s.maxCubemapSize):Et[Q]=at?v.image[Q].image:v.image[Q],Et[Q]=Ut(v,Et[Q]);let Nt=Et[0],Bt=r.convert(v.format,v.colorSpace),bt=r.convert(v.type),$t=M(v.internalFormat,Bt,bt,v.colorSpace),Xt=v.isVideoTexture!==!0,oe=J.__version===void 0||$===!0,U=et.dataReady,pt=D(v,Nt);zt(n.TEXTURE_CUBE_MAP,v);let q;if(Zt){Xt&&oe&&e.texStorage2D(n.TEXTURE_CUBE_MAP,pt,$t,Nt.width,Nt.height);for(let Q=0;Q<6;Q++){q=Et[Q].mipmaps;for(let _t=0;_t<q.length;_t++){let mt=q[_t];v.format!==sn?Bt!==null?Xt?U&&e.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t,0,0,mt.width,mt.height,Bt,mt.data):e.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t,$t,mt.width,mt.height,0,mt.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):Xt?U&&e.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t,0,0,mt.width,mt.height,Bt,bt,mt.data):e.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t,$t,mt.width,mt.height,0,Bt,bt,mt.data)}}}else{if(q=v.mipmaps,Xt&&oe){q.length>0&&pt++;let Q=xt(Et[0]);e.texStorage2D(n.TEXTURE_CUBE_MAP,pt,$t,Q.width,Q.height)}for(let Q=0;Q<6;Q++)if(at){Xt?U&&e.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,0,0,0,Et[Q].width,Et[Q].height,Bt,bt,Et[Q].data):e.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,0,$t,Et[Q].width,Et[Q].height,0,Bt,bt,Et[Q].data);for(let _t=0;_t<q.length;_t++){let Vt=q[_t].image[Q].image;Xt?U&&e.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t+1,0,0,Vt.width,Vt.height,Bt,bt,Vt.data):e.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t+1,$t,Vt.width,Vt.height,0,Bt,bt,Vt.data)}}else{Xt?U&&e.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,0,0,0,Bt,bt,Et[Q]):e.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,0,$t,Bt,bt,Et[Q]);for(let _t=0;_t<q.length;_t++){let mt=q[_t];Xt?U&&e.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t+1,0,0,Bt,bt,mt.image[Q]):e.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+Q,_t+1,$t,Bt,bt,mt.image[Q])}}}m(v)&&p(n.TEXTURE_CUBE_MAP),J.__version=et.version,v.onUpdate&&v.onUpdate(v)}w.__version=v.version}function nt(w,v,z,$,et,J){let Tt=r.convert(z.format,z.colorSpace),ft=r.convert(z.type),yt=M(z.internalFormat,Tt,ft,z.colorSpace),Zt=i.get(v),at=i.get(z);if(at.__renderTarget=v,!Zt.__hasExternalTextures){let Et=Math.max(1,v.width>>J),Nt=Math.max(1,v.height>>J);et===n.TEXTURE_3D||et===n.TEXTURE_2D_ARRAY?e.texImage3D(et,J,yt,Et,Nt,v.depth,0,Tt,ft,null):e.texImage2D(et,J,yt,Et,Nt,0,Tt,ft,null)}e.bindFramebuffer(n.FRAMEBUFFER,w),Mt(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,$,et,at.__webglTexture,0,ot(v)):(et===n.TEXTURE_2D||et>=n.TEXTURE_CUBE_MAP_POSITIVE_X&&et<=n.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&n.framebufferTexture2D(n.FRAMEBUFFER,$,et,at.__webglTexture,J),e.bindFramebuffer(n.FRAMEBUFFER,null)}function tt(w,v,z){if(n.bindRenderbuffer(n.RENDERBUFFER,w),v.depthBuffer){let $=v.depthTexture,et=$&&$.isDepthTexture?$.type:null,J=x(v.stencilBuffer,et),Tt=v.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,ft=ot(v);Mt(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,ft,J,v.width,v.height):z?n.renderbufferStorageMultisample(n.RENDERBUFFER,ft,J,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,J,v.width,v.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,Tt,n.RENDERBUFFER,w)}else{let $=v.textures;for(let et=0;et<$.length;et++){let J=$[et],Tt=r.convert(J.format,J.colorSpace),ft=r.convert(J.type),yt=M(J.internalFormat,Tt,ft,J.colorSpace),Zt=ot(v);z&&Mt(v)===!1?n.renderbufferStorageMultisample(n.RENDERBUFFER,Zt,yt,v.width,v.height):Mt(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,Zt,yt,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,yt,v.width,v.height)}}n.bindRenderbuffer(n.RENDERBUFFER,null)}function wt(w,v){if(v&&v.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(e.bindFramebuffer(n.FRAMEBUFFER,w),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");let $=i.get(v.depthTexture);$.__renderTarget=v,(!$.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),Y(v.depthTexture,0);let et=$.__webglTexture,J=ot(v);if(v.depthTexture.format===$i)Mt(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,n.DEPTH_ATTACHMENT,n.TEXTURE_2D,et,0,J):n.framebufferTexture2D(n.FRAMEBUFFER,n.DEPTH_ATTACHMENT,n.TEXTURE_2D,et,0);else if(v.depthTexture.format===ns)Mt(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,n.DEPTH_STENCIL_ATTACHMENT,n.TEXTURE_2D,et,0,J):n.framebufferTexture2D(n.FRAMEBUFFER,n.DEPTH_STENCIL_ATTACHMENT,n.TEXTURE_2D,et,0);else throw new Error("Unknown depthTexture format")}function Dt(w){let v=i.get(w),z=w.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==w.depthTexture){let $=w.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),$){let et=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,$.removeEventListener("dispose",et)};$.addEventListener("dispose",et),v.__depthDisposeCallback=et}v.__boundDepthTexture=$}if(w.depthTexture&&!v.__autoAllocateDepthBuffer){if(z)throw new Error("target.depthTexture not supported in Cube render targets");wt(v.__webglFramebuffer,w)}else if(z){v.__webglDepthbuffer=[];for(let $=0;$<6;$++)if(e.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer[$]),v.__webglDepthbuffer[$]===void 0)v.__webglDepthbuffer[$]=n.createRenderbuffer(),tt(v.__webglDepthbuffer[$],w,!1);else{let et=w.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,J=v.__webglDepthbuffer[$];n.bindRenderbuffer(n.RENDERBUFFER,J),n.framebufferRenderbuffer(n.FRAMEBUFFER,et,n.RENDERBUFFER,J)}}else if(e.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=n.createRenderbuffer(),tt(v.__webglDepthbuffer,w,!1);else{let $=w.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,et=v.__webglDepthbuffer;n.bindRenderbuffer(n.RENDERBUFFER,et),n.framebufferRenderbuffer(n.FRAMEBUFFER,$,n.RENDERBUFFER,et)}e.bindFramebuffer(n.FRAMEBUFFER,null)}function Lt(w,v,z){let $=i.get(w);v!==void 0&&nt($.__webglFramebuffer,w,w.texture,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,0),z!==void 0&&Dt(w)}function Yt(w){let v=w.texture,z=i.get(w),$=i.get(v);w.addEventListener("dispose",R);let et=w.textures,J=w.isWebGLCubeRenderTarget===!0,Tt=et.length>1;if(Tt||($.__webglTexture===void 0&&($.__webglTexture=n.createTexture()),$.__version=v.version,o.memory.textures++),J){z.__webglFramebuffer=[];for(let ft=0;ft<6;ft++)if(v.mipmaps&&v.mipmaps.length>0){z.__webglFramebuffer[ft]=[];for(let yt=0;yt<v.mipmaps.length;yt++)z.__webglFramebuffer[ft][yt]=n.createFramebuffer()}else z.__webglFramebuffer[ft]=n.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){z.__webglFramebuffer=[];for(let ft=0;ft<v.mipmaps.length;ft++)z.__webglFramebuffer[ft]=n.createFramebuffer()}else z.__webglFramebuffer=n.createFramebuffer();if(Tt)for(let ft=0,yt=et.length;ft<yt;ft++){let Zt=i.get(et[ft]);Zt.__webglTexture===void 0&&(Zt.__webglTexture=n.createTexture(),o.memory.textures++)}if(w.samples>0&&Mt(w)===!1){z.__webglMultisampledFramebuffer=n.createFramebuffer(),z.__webglColorRenderbuffer=[],e.bindFramebuffer(n.FRAMEBUFFER,z.__webglMultisampledFramebuffer);for(let ft=0;ft<et.length;ft++){let yt=et[ft];z.__webglColorRenderbuffer[ft]=n.createRenderbuffer(),n.bindRenderbuffer(n.RENDERBUFFER,z.__webglColorRenderbuffer[ft]);let Zt=r.convert(yt.format,yt.colorSpace),at=r.convert(yt.type),Et=M(yt.internalFormat,Zt,at,yt.colorSpace,w.isXRRenderTarget===!0),Nt=ot(w);n.renderbufferStorageMultisample(n.RENDERBUFFER,Nt,Et,w.width,w.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+ft,n.RENDERBUFFER,z.__webglColorRenderbuffer[ft])}n.bindRenderbuffer(n.RENDERBUFFER,null),w.depthBuffer&&(z.__webglDepthRenderbuffer=n.createRenderbuffer(),tt(z.__webglDepthRenderbuffer,w,!0)),e.bindFramebuffer(n.FRAMEBUFFER,null)}}if(J){e.bindTexture(n.TEXTURE_CUBE_MAP,$.__webglTexture),zt(n.TEXTURE_CUBE_MAP,v);for(let ft=0;ft<6;ft++)if(v.mipmaps&&v.mipmaps.length>0)for(let yt=0;yt<v.mipmaps.length;yt++)nt(z.__webglFramebuffer[ft][yt],w,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+ft,yt);else nt(z.__webglFramebuffer[ft],w,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+ft,0);m(v)&&p(n.TEXTURE_CUBE_MAP),e.unbindTexture()}else if(Tt){for(let ft=0,yt=et.length;ft<yt;ft++){let Zt=et[ft],at=i.get(Zt);e.bindTexture(n.TEXTURE_2D,at.__webglTexture),zt(n.TEXTURE_2D,Zt),nt(z.__webglFramebuffer,w,Zt,n.COLOR_ATTACHMENT0+ft,n.TEXTURE_2D,0),m(Zt)&&p(n.TEXTURE_2D)}e.unbindTexture()}else{let ft=n.TEXTURE_2D;if((w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(ft=w.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),e.bindTexture(ft,$.__webglTexture),zt(ft,v),v.mipmaps&&v.mipmaps.length>0)for(let yt=0;yt<v.mipmaps.length;yt++)nt(z.__webglFramebuffer[yt],w,v,n.COLOR_ATTACHMENT0,ft,yt);else nt(z.__webglFramebuffer,w,v,n.COLOR_ATTACHMENT0,ft,0);m(v)&&p(ft),e.unbindTexture()}w.depthBuffer&&Dt(w)}function Z(w){let v=w.textures;for(let z=0,$=v.length;z<$;z++){let et=v[z];if(m(et)){let J=S(w),Tt=i.get(et).__webglTexture;e.bindTexture(J,Tt),p(J),e.unbindTexture()}}}let st=[],A=[];function St(w){if(w.samples>0){if(Mt(w)===!1){let v=w.textures,z=w.width,$=w.height,et=n.COLOR_BUFFER_BIT,J=w.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,Tt=i.get(w),ft=v.length>1;if(ft)for(let yt=0;yt<v.length;yt++)e.bindFramebuffer(n.FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+yt,n.RENDERBUFFER,null),e.bindFramebuffer(n.FRAMEBUFFER,Tt.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+yt,n.TEXTURE_2D,null,0);e.bindFramebuffer(n.READ_FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),e.bindFramebuffer(n.DRAW_FRAMEBUFFER,Tt.__webglFramebuffer);for(let yt=0;yt<v.length;yt++){if(w.resolveDepthBuffer&&(w.depthBuffer&&(et|=n.DEPTH_BUFFER_BIT),w.stencilBuffer&&w.resolveStencilBuffer&&(et|=n.STENCIL_BUFFER_BIT)),ft){n.framebufferRenderbuffer(n.READ_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.RENDERBUFFER,Tt.__webglColorRenderbuffer[yt]);let Zt=i.get(v[yt]).__webglTexture;n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,Zt,0)}n.blitFramebuffer(0,0,z,$,0,0,z,$,et,n.NEAREST),c===!0&&(st.length=0,A.length=0,st.push(n.COLOR_ATTACHMENT0+yt),w.depthBuffer&&w.resolveDepthBuffer===!1&&(st.push(J),A.push(J),n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,A)),n.invalidateFramebuffer(n.READ_FRAMEBUFFER,st))}if(e.bindFramebuffer(n.READ_FRAMEBUFFER,null),e.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),ft)for(let yt=0;yt<v.length;yt++){e.bindFramebuffer(n.FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+yt,n.RENDERBUFFER,Tt.__webglColorRenderbuffer[yt]);let Zt=i.get(v[yt]).__webglTexture;e.bindFramebuffer(n.FRAMEBUFFER,Tt.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+yt,n.TEXTURE_2D,Zt,0)}e.bindFramebuffer(n.DRAW_FRAMEBUFFER,Tt.__webglMultisampledFramebuffer)}else if(w.depthBuffer&&w.resolveDepthBuffer===!1&&c){let v=w.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,[v])}}}function ot(w){return Math.min(s.maxSamples,w.samples)}function Mt(w){let v=i.get(w);return w.samples>0&&t.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function ht(w){let v=o.render.frame;h.get(w)!==v&&(h.set(w,v),w.update())}function Ut(w,v){let z=w.colorSpace,$=w.format,et=w.type;return w.isCompressedTexture===!0||w.isVideoTexture===!0||z!==cs&&z!==Xn&&(Jt.getTransfer(z)===ne?($!==sn||et!==Pn)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",z)),v}function xt(w){return typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement?(l.width=w.naturalWidth||w.width,l.height=w.naturalHeight||w.height):typeof VideoFrame<"u"&&w instanceof VideoFrame?(l.width=w.displayWidth,l.height=w.displayHeight):(l.width=w.width,l.height=w.height),l}this.allocateTextureUnit=N,this.resetTextureUnits=B,this.setTexture2D=Y,this.setTexture2DArray=V,this.setTexture3D=rt,this.setTextureCube=X,this.rebindTextures=Lt,this.setupRenderTarget=Yt,this.updateRenderTargetMipmap=Z,this.updateMultisampleRenderTarget=St,this.setupDepthRenderbuffer=Dt,this.setupFrameBufferTexture=nt,this.useMultisampledRTT=Mt}function E0(n,t){function e(i,s=Xn){let r,o=Jt.getTransfer(s);if(i===Pn)return n.UNSIGNED_BYTE;if(i===Ac)return n.UNSIGNED_SHORT_4_4_4_4;if(i===Rc)return n.UNSIGNED_SHORT_5_5_5_1;if(i===Eh)return n.UNSIGNED_INT_5_9_9_9_REV;if(i===Mh)return n.BYTE;if(i===Sh)return n.SHORT;if(i===Us)return n.UNSIGNED_SHORT;if(i===Tc)return n.INT;if(i===Mi)return n.UNSIGNED_INT;if(i===Rn)return n.FLOAT;if(i===Gs)return n.HALF_FLOAT;if(i===bh)return n.ALPHA;if(i===wh)return n.RGB;if(i===sn)return n.RGBA;if(i===Th)return n.LUMINANCE;if(i===Ah)return n.LUMINANCE_ALPHA;if(i===$i)return n.DEPTH_COMPONENT;if(i===ns)return n.DEPTH_STENCIL;if(i===Rh)return n.RED;if(i===Cc)return n.RED_INTEGER;if(i===Ch)return n.RG;if(i===Ic)return n.RG_INTEGER;if(i===Pc)return n.RGBA_INTEGER;if(i===Sr||i===Er||i===br||i===wr)if(o===ne)if(r=t.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===Sr)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===Er)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===br)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===wr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=t.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===Sr)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===Er)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===br)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===wr)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===ha||i===ua||i===da||i===fa)if(r=t.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===ha)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===ua)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===da)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===fa)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===pa||i===ma||i===ga)if(r=t.get("WEBGL_compressed_texture_etc"),r!==null){if(i===pa||i===ma)return o===ne?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===ga)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(i===_a||i===xa||i===ya||i===va||i===Ma||i===Sa||i===Ea||i===ba||i===wa||i===Ta||i===Aa||i===Ra||i===Ca||i===Ia)if(r=t.get("WEBGL_compressed_texture_astc"),r!==null){if(i===_a)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===xa)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===ya)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===va)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===Ma)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===Sa)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===Ea)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===ba)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===wa)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===Ta)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===Aa)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===Ra)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===Ca)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===Ia)return o===ne?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===Tr||i===Pa||i===La)if(r=t.get("EXT_texture_compression_bptc"),r!==null){if(i===Tr)return o===ne?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===Pa)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===La)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===Ih||i===Da||i===Ua||i===Na)if(r=t.get("EXT_texture_compression_rgtc"),r!==null){if(i===Tr)return r.COMPRESSED_RED_RGTC1_EXT;if(i===Da)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===Ua)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===Na)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===es?n.UNSIGNED_INT_24_8:n[i]!==void 0?n[i]:null}return{convert:e}}var Ja=class extends Pe{constructor(t=[]){super(),this.isArrayCamera=!0,this.cameras=t}},Ot=class extends Ue{constructor(){super(),this.isGroup=!0,this.type="Group"}},b0={type:"move"},Is=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Ot,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Ot,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new L,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new L),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Ot,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new L,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new L),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){let e=this._hand;if(e)for(let i of t.hand.values())this._getHandJoint(e,i)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,i){let s=null,r=null,o=null,a=this._targetRay,c=this._grip,l=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(l&&t.hand){o=!0;for(let _ of t.hand.values()){let m=e.getJointPose(_,i),p=this._getHandJoint(l,_);m!==null&&(p.matrix.fromArray(m.transform.matrix),p.matrix.decompose(p.position,p.rotation,p.scale),p.matrixWorldNeedsUpdate=!0,p.jointRadius=m.radius),p.visible=m!==null}let h=l.joints["index-finger-tip"],u=l.joints["thumb-tip"],d=h.position.distanceTo(u.position),f=.02,g=.005;l.inputState.pinching&&d>f+g?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!l.inputState.pinching&&d<=f-g&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else c!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,i),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1));a!==null&&(s=e.getPose(t.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(b0)))}return a!==null&&(a.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=o!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){let i=new Ot;i.matrixAutoUpdate=!1,i.visible=!1,t.joints[e.jointName]=i,t.add(i)}return t.joints[e.jointName]}},w0=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,T0=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,Ka=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(t,e,i){if(this.texture===null){let s=new We,r=t.properties.get(s);r.__webglTexture=e.texture,(e.depthNear!=i.depthNear||e.depthFar!=i.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=s}}getMesh(t){if(this.texture!==null&&this.mesh===null){let e=t.cameras[0].viewport,i=new dn({vertexShader:w0,fragmentShader:T0,uniforms:{depthColor:{value:this.texture},depthWidth:{value:e.z},depthHeight:{value:e.w}}});this.mesh=new Pt(new Un(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},Qa=class extends Jn{constructor(t,e){super();let i=this,s=null,r=1,o=null,a="local-floor",c=1,l=null,h=null,u=null,d=null,f=null,g=null,_=new Ka,m=e.getContextAttributes(),p=null,S=null,M=[],x=[],D=new j,T=null,R=new Pe;R.viewport=new ie;let C=new Pe;C.viewport=new ie;let b=[R,C],y=new Ja,I=null,B=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(P){let k=M[P];return k===void 0&&(k=new Is,M[P]=k),k.getTargetRaySpace()},this.getControllerGrip=function(P){let k=M[P];return k===void 0&&(k=new Is,M[P]=k),k.getGripSpace()},this.getHand=function(P){let k=M[P];return k===void 0&&(k=new Is,M[P]=k),k.getHandSpace()};function N(P){let k=x.indexOf(P.inputSource);if(k===-1)return;let nt=M[k];nt!==void 0&&(nt.update(P.inputSource,P.frame,l||o),nt.dispatchEvent({type:P.type,data:P.inputSource}))}function H(){s.removeEventListener("select",N),s.removeEventListener("selectstart",N),s.removeEventListener("selectend",N),s.removeEventListener("squeeze",N),s.removeEventListener("squeezestart",N),s.removeEventListener("squeezeend",N),s.removeEventListener("end",H),s.removeEventListener("inputsourceschange",Y);for(let P=0;P<M.length;P++){let k=x[P];k!==null&&(x[P]=null,M[P].disconnect(k))}I=null,B=null,_.reset(),t.setRenderTarget(p),f=null,d=null,u=null,s=null,S=null,Kt.stop(),i.isPresenting=!1,t.setPixelRatio(T),t.setSize(D.width,D.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(P){r=P,i.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(P){a=P,i.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||o},this.setReferenceSpace=function(P){l=P},this.getBaseLayer=function(){return d!==null?d:f},this.getBinding=function(){return u},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(P){if(s=P,s!==null){if(p=t.getRenderTarget(),s.addEventListener("select",N),s.addEventListener("selectstart",N),s.addEventListener("selectend",N),s.addEventListener("squeeze",N),s.addEventListener("squeezestart",N),s.addEventListener("squeezeend",N),s.addEventListener("end",H),s.addEventListener("inputsourceschange",Y),m.xrCompatible!==!0&&await e.makeXRCompatible(),T=t.getPixelRatio(),t.getSize(D),s.renderState.layers===void 0){let k={antialias:m.antialias,alpha:!0,depth:m.depth,stencil:m.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,e,k),s.updateRenderState({baseLayer:f}),t.setPixelRatio(1),t.setSize(f.framebufferWidth,f.framebufferHeight,!1),S=new Ln(f.framebufferWidth,f.framebufferHeight,{format:sn,type:Pn,colorSpace:t.outputColorSpace,stencilBuffer:m.stencil})}else{let k=null,nt=null,tt=null;m.depth&&(tt=m.stencil?e.DEPTH24_STENCIL8:e.DEPTH_COMPONENT24,k=m.stencil?ns:$i,nt=m.stencil?es:Mi);let wt={colorFormat:e.RGBA8,depthFormat:tt,scaleFactor:r};u=new XRWebGLBinding(s,e),d=u.createProjectionLayer(wt),s.updateRenderState({layers:[d]}),t.setPixelRatio(1),t.setSize(d.textureWidth,d.textureHeight,!1),S=new Ln(d.textureWidth,d.textureHeight,{format:sn,type:Pn,depthTexture:new Hr(d.textureWidth,d.textureHeight,nt,void 0,void 0,void 0,void 0,void 0,void 0,k),stencilBuffer:m.stencil,colorSpace:t.outputColorSpace,samples:m.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1})}S.isXRRenderTarget=!0,this.setFoveation(c),l=null,o=await s.requestReferenceSpace(a),Kt.setContext(s),Kt.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return _.getDepthTexture()};function Y(P){for(let k=0;k<P.removed.length;k++){let nt=P.removed[k],tt=x.indexOf(nt);tt>=0&&(x[tt]=null,M[tt].disconnect(nt))}for(let k=0;k<P.added.length;k++){let nt=P.added[k],tt=x.indexOf(nt);if(tt===-1){for(let Dt=0;Dt<M.length;Dt++)if(Dt>=x.length){x.push(nt),tt=Dt;break}else if(x[Dt]===null){x[Dt]=nt,tt=Dt;break}if(tt===-1)break}let wt=M[tt];wt&&wt.connect(nt)}}let V=new L,rt=new L;function X(P,k,nt){V.setFromMatrixPosition(k.matrixWorld),rt.setFromMatrixPosition(nt.matrixWorld);let tt=V.distanceTo(rt),wt=k.projectionMatrix.elements,Dt=nt.projectionMatrix.elements,Lt=wt[14]/(wt[10]-1),Yt=wt[14]/(wt[10]+1),Z=(wt[9]+1)/wt[5],st=(wt[9]-1)/wt[5],A=(wt[8]-1)/wt[0],St=(Dt[8]+1)/Dt[0],ot=Lt*A,Mt=Lt*St,ht=tt/(-A+St),Ut=ht*-A;if(k.matrixWorld.decompose(P.position,P.quaternion,P.scale),P.translateX(Ut),P.translateZ(ht),P.matrixWorld.compose(P.position,P.quaternion,P.scale),P.matrixWorldInverse.copy(P.matrixWorld).invert(),wt[10]===-1)P.projectionMatrix.copy(k.projectionMatrix),P.projectionMatrixInverse.copy(k.projectionMatrixInverse);else{let xt=Lt+ht,w=Yt+ht,v=ot-Ut,z=Mt+(tt-Ut),$=Z*Yt/w*xt,et=st*Yt/w*xt;P.projectionMatrix.makePerspective(v,z,$,et,xt,w),P.projectionMatrixInverse.copy(P.projectionMatrix).invert()}}function it(P,k){k===null?P.matrixWorld.copy(P.matrix):P.matrixWorld.multiplyMatrices(k.matrixWorld,P.matrix),P.matrixWorldInverse.copy(P.matrixWorld).invert()}this.updateCamera=function(P){if(s===null)return;let k=P.near,nt=P.far;_.texture!==null&&(_.depthNear>0&&(k=_.depthNear),_.depthFar>0&&(nt=_.depthFar)),y.near=C.near=R.near=k,y.far=C.far=R.far=nt,(I!==y.near||B!==y.far)&&(s.updateRenderState({depthNear:y.near,depthFar:y.far}),I=y.near,B=y.far),R.layers.mask=P.layers.mask|2,C.layers.mask=P.layers.mask|4,y.layers.mask=R.layers.mask|C.layers.mask;let tt=P.parent,wt=y.cameras;it(y,tt);for(let Dt=0;Dt<wt.length;Dt++)it(wt[Dt],tt);wt.length===2?X(y,R,C):y.projectionMatrix.copy(R.projectionMatrix),dt(P,y,tt)};function dt(P,k,nt){nt===null?P.matrix.copy(k.matrixWorld):(P.matrix.copy(nt.matrixWorld),P.matrix.invert(),P.matrix.multiply(k.matrixWorld)),P.matrix.decompose(P.position,P.quaternion,P.scale),P.updateMatrixWorld(!0),P.projectionMatrix.copy(k.projectionMatrix),P.projectionMatrixInverse.copy(k.projectionMatrixInverse),P.isPerspectiveCamera&&(P.fov=Ns*2*Math.atan(1/P.projectionMatrix.elements[5]),P.zoom=1)}this.getCamera=function(){return y},this.getFoveation=function(){if(!(d===null&&f===null))return c},this.setFoveation=function(P){c=P,d!==null&&(d.fixedFoveation=P),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=P)},this.hasDepthSensing=function(){return _.texture!==null},this.getDepthSensingMesh=function(){return _.getMesh(y)};let vt=null;function zt(P,k){if(h=k.getViewerPose(l||o),g=k,h!==null){let nt=h.views;f!==null&&(t.setRenderTargetFramebuffer(S,f.framebuffer),t.setRenderTarget(S));let tt=!1;nt.length!==y.cameras.length&&(y.cameras.length=0,tt=!0);for(let Dt=0;Dt<nt.length;Dt++){let Lt=nt[Dt],Yt=null;if(f!==null)Yt=f.getViewport(Lt);else{let st=u.getViewSubImage(d,Lt);Yt=st.viewport,Dt===0&&(t.setRenderTargetTextures(S,st.colorTexture,d.ignoreDepthValues?void 0:st.depthStencilTexture),t.setRenderTarget(S))}let Z=b[Dt];Z===void 0&&(Z=new Pe,Z.layers.enable(Dt),Z.viewport=new ie,b[Dt]=Z),Z.matrix.fromArray(Lt.transform.matrix),Z.matrix.decompose(Z.position,Z.quaternion,Z.scale),Z.projectionMatrix.fromArray(Lt.projectionMatrix),Z.projectionMatrixInverse.copy(Z.projectionMatrix).invert(),Z.viewport.set(Yt.x,Yt.y,Yt.width,Yt.height),Dt===0&&(y.matrix.copy(Z.matrix),y.matrix.decompose(y.position,y.quaternion,y.scale)),tt===!0&&y.cameras.push(Z)}let wt=s.enabledFeatures;if(wt&&wt.includes("depth-sensing")){let Dt=u.getDepthInformation(nt[0]);Dt&&Dt.isValid&&Dt.texture&&_.init(t,Dt,s.renderState)}}for(let nt=0;nt<M.length;nt++){let tt=x[nt],wt=M[nt];tt!==null&&wt!==void 0&&wt.update(tt,k,l||o)}vt&&vt(P,k),k.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:k}),g=null}let Kt=new Fh;Kt.setAnimationLoop(zt),this.setAnimationLoop=function(P){vt=P},this.dispose=function(){}}},pi=new un,A0=new de;function R0(n,t){function e(m,p){m.matrixAutoUpdate===!0&&m.updateMatrix(),p.value.copy(m.matrix)}function i(m,p){p.color.getRGB(m.fogColor.value,Nh(n)),p.isFog?(m.fogNear.value=p.near,m.fogFar.value=p.far):p.isFogExp2&&(m.fogDensity.value=p.density)}function s(m,p,S,M,x){p.isMeshBasicMaterial||p.isMeshLambertMaterial?r(m,p):p.isMeshToonMaterial?(r(m,p),u(m,p)):p.isMeshPhongMaterial?(r(m,p),h(m,p)):p.isMeshStandardMaterial?(r(m,p),d(m,p),p.isMeshPhysicalMaterial&&f(m,p,x)):p.isMeshMatcapMaterial?(r(m,p),g(m,p)):p.isMeshDepthMaterial?r(m,p):p.isMeshDistanceMaterial?(r(m,p),_(m,p)):p.isMeshNormalMaterial?r(m,p):p.isLineBasicMaterial?(o(m,p),p.isLineDashedMaterial&&a(m,p)):p.isPointsMaterial?c(m,p,S,M):p.isSpriteMaterial?l(m,p):p.isShadowMaterial?(m.color.value.copy(p.color),m.opacity.value=p.opacity):p.isShaderMaterial&&(p.uniformsNeedUpdate=!1)}function r(m,p){m.opacity.value=p.opacity,p.color&&m.diffuse.value.copy(p.color),p.emissive&&m.emissive.value.copy(p.emissive).multiplyScalar(p.emissiveIntensity),p.map&&(m.map.value=p.map,e(p.map,m.mapTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,e(p.alphaMap,m.alphaMapTransform)),p.bumpMap&&(m.bumpMap.value=p.bumpMap,e(p.bumpMap,m.bumpMapTransform),m.bumpScale.value=p.bumpScale,p.side===Be&&(m.bumpScale.value*=-1)),p.normalMap&&(m.normalMap.value=p.normalMap,e(p.normalMap,m.normalMapTransform),m.normalScale.value.copy(p.normalScale),p.side===Be&&m.normalScale.value.negate()),p.displacementMap&&(m.displacementMap.value=p.displacementMap,e(p.displacementMap,m.displacementMapTransform),m.displacementScale.value=p.displacementScale,m.displacementBias.value=p.displacementBias),p.emissiveMap&&(m.emissiveMap.value=p.emissiveMap,e(p.emissiveMap,m.emissiveMapTransform)),p.specularMap&&(m.specularMap.value=p.specularMap,e(p.specularMap,m.specularMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest);let S=t.get(p),M=S.envMap,x=S.envMapRotation;M&&(m.envMap.value=M,pi.copy(x),pi.x*=-1,pi.y*=-1,pi.z*=-1,M.isCubeTexture&&M.isRenderTargetTexture===!1&&(pi.y*=-1,pi.z*=-1),m.envMapRotation.value.setFromMatrix4(A0.makeRotationFromEuler(pi)),m.flipEnvMap.value=M.isCubeTexture&&M.isRenderTargetTexture===!1?-1:1,m.reflectivity.value=p.reflectivity,m.ior.value=p.ior,m.refractionRatio.value=p.refractionRatio),p.lightMap&&(m.lightMap.value=p.lightMap,m.lightMapIntensity.value=p.lightMapIntensity,e(p.lightMap,m.lightMapTransform)),p.aoMap&&(m.aoMap.value=p.aoMap,m.aoMapIntensity.value=p.aoMapIntensity,e(p.aoMap,m.aoMapTransform))}function o(m,p){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,p.map&&(m.map.value=p.map,e(p.map,m.mapTransform))}function a(m,p){m.dashSize.value=p.dashSize,m.totalSize.value=p.dashSize+p.gapSize,m.scale.value=p.scale}function c(m,p,S,M){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,m.size.value=p.size*S,m.scale.value=M*.5,p.map&&(m.map.value=p.map,e(p.map,m.uvTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,e(p.alphaMap,m.alphaMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest)}function l(m,p){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,m.rotation.value=p.rotation,p.map&&(m.map.value=p.map,e(p.map,m.mapTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,e(p.alphaMap,m.alphaMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest)}function h(m,p){m.specular.value.copy(p.specular),m.shininess.value=Math.max(p.shininess,1e-4)}function u(m,p){p.gradientMap&&(m.gradientMap.value=p.gradientMap)}function d(m,p){m.metalness.value=p.metalness,p.metalnessMap&&(m.metalnessMap.value=p.metalnessMap,e(p.metalnessMap,m.metalnessMapTransform)),m.roughness.value=p.roughness,p.roughnessMap&&(m.roughnessMap.value=p.roughnessMap,e(p.roughnessMap,m.roughnessMapTransform)),p.envMap&&(m.envMapIntensity.value=p.envMapIntensity)}function f(m,p,S){m.ior.value=p.ior,p.sheen>0&&(m.sheenColor.value.copy(p.sheenColor).multiplyScalar(p.sheen),m.sheenRoughness.value=p.sheenRoughness,p.sheenColorMap&&(m.sheenColorMap.value=p.sheenColorMap,e(p.sheenColorMap,m.sheenColorMapTransform)),p.sheenRoughnessMap&&(m.sheenRoughnessMap.value=p.sheenRoughnessMap,e(p.sheenRoughnessMap,m.sheenRoughnessMapTransform))),p.clearcoat>0&&(m.clearcoat.value=p.clearcoat,m.clearcoatRoughness.value=p.clearcoatRoughness,p.clearcoatMap&&(m.clearcoatMap.value=p.clearcoatMap,e(p.clearcoatMap,m.clearcoatMapTransform)),p.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=p.clearcoatRoughnessMap,e(p.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),p.clearcoatNormalMap&&(m.clearcoatNormalMap.value=p.clearcoatNormalMap,e(p.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(p.clearcoatNormalScale),p.side===Be&&m.clearcoatNormalScale.value.negate())),p.dispersion>0&&(m.dispersion.value=p.dispersion),p.iridescence>0&&(m.iridescence.value=p.iridescence,m.iridescenceIOR.value=p.iridescenceIOR,m.iridescenceThicknessMinimum.value=p.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=p.iridescenceThicknessRange[1],p.iridescenceMap&&(m.iridescenceMap.value=p.iridescenceMap,e(p.iridescenceMap,m.iridescenceMapTransform)),p.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=p.iridescenceThicknessMap,e(p.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),p.transmission>0&&(m.transmission.value=p.transmission,m.transmissionSamplerMap.value=S.texture,m.transmissionSamplerSize.value.set(S.width,S.height),p.transmissionMap&&(m.transmissionMap.value=p.transmissionMap,e(p.transmissionMap,m.transmissionMapTransform)),m.thickness.value=p.thickness,p.thicknessMap&&(m.thicknessMap.value=p.thicknessMap,e(p.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=p.attenuationDistance,m.attenuationColor.value.copy(p.attenuationColor)),p.anisotropy>0&&(m.anisotropyVector.value.set(p.anisotropy*Math.cos(p.anisotropyRotation),p.anisotropy*Math.sin(p.anisotropyRotation)),p.anisotropyMap&&(m.anisotropyMap.value=p.anisotropyMap,e(p.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=p.specularIntensity,m.specularColor.value.copy(p.specularColor),p.specularColorMap&&(m.specularColorMap.value=p.specularColorMap,e(p.specularColorMap,m.specularColorMapTransform)),p.specularIntensityMap&&(m.specularIntensityMap.value=p.specularIntensityMap,e(p.specularIntensityMap,m.specularIntensityMapTransform))}function g(m,p){p.matcap&&(m.matcap.value=p.matcap)}function _(m,p){let S=t.get(p).light;m.referencePosition.value.setFromMatrixPosition(S.matrixWorld),m.nearDistance.value=S.shadow.camera.near,m.farDistance.value=S.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function C0(n,t,e,i){let s={},r={},o=[],a=n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);function c(S,M){let x=M.program;i.uniformBlockBinding(S,x)}function l(S,M){let x=s[S.id];x===void 0&&(g(S),x=h(S),s[S.id]=x,S.addEventListener("dispose",m));let D=M.program;i.updateUBOMapping(S,D);let T=t.render.frame;r[S.id]!==T&&(d(S),r[S.id]=T)}function h(S){let M=u();S.__bindingPointIndex=M;let x=n.createBuffer(),D=S.__size,T=S.usage;return n.bindBuffer(n.UNIFORM_BUFFER,x),n.bufferData(n.UNIFORM_BUFFER,D,T),n.bindBuffer(n.UNIFORM_BUFFER,null),n.bindBufferBase(n.UNIFORM_BUFFER,M,x),x}function u(){for(let S=0;S<a;S++)if(o.indexOf(S)===-1)return o.push(S),S;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(S){let M=s[S.id],x=S.uniforms,D=S.__cache;n.bindBuffer(n.UNIFORM_BUFFER,M);for(let T=0,R=x.length;T<R;T++){let C=Array.isArray(x[T])?x[T]:[x[T]];for(let b=0,y=C.length;b<y;b++){let I=C[b];if(f(I,T,b,D)===!0){let B=I.__offset,N=Array.isArray(I.value)?I.value:[I.value],H=0;for(let Y=0;Y<N.length;Y++){let V=N[Y],rt=_(V);typeof V=="number"||typeof V=="boolean"?(I.__data[0]=V,n.bufferSubData(n.UNIFORM_BUFFER,B+H,I.__data)):V.isMatrix3?(I.__data[0]=V.elements[0],I.__data[1]=V.elements[1],I.__data[2]=V.elements[2],I.__data[3]=0,I.__data[4]=V.elements[3],I.__data[5]=V.elements[4],I.__data[6]=V.elements[5],I.__data[7]=0,I.__data[8]=V.elements[6],I.__data[9]=V.elements[7],I.__data[10]=V.elements[8],I.__data[11]=0):(V.toArray(I.__data,H),H+=rt.storage/Float32Array.BYTES_PER_ELEMENT)}n.bufferSubData(n.UNIFORM_BUFFER,B,I.__data)}}}n.bindBuffer(n.UNIFORM_BUFFER,null)}function f(S,M,x,D){let T=S.value,R=M+"_"+x;if(D[R]===void 0)return typeof T=="number"||typeof T=="boolean"?D[R]=T:D[R]=T.clone(),!0;{let C=D[R];if(typeof T=="number"||typeof T=="boolean"){if(C!==T)return D[R]=T,!0}else if(C.equals(T)===!1)return C.copy(T),!0}return!1}function g(S){let M=S.uniforms,x=0,D=16;for(let R=0,C=M.length;R<C;R++){let b=Array.isArray(M[R])?M[R]:[M[R]];for(let y=0,I=b.length;y<I;y++){let B=b[y],N=Array.isArray(B.value)?B.value:[B.value];for(let H=0,Y=N.length;H<Y;H++){let V=N[H],rt=_(V),X=x%D,it=X%rt.boundary,dt=X+it;x+=it,dt!==0&&D-dt<rt.storage&&(x+=D-dt),B.__data=new Float32Array(rt.storage/Float32Array.BYTES_PER_ELEMENT),B.__offset=x,x+=rt.storage}}}let T=x%D;return T>0&&(x+=D-T),S.__size=x,S.__cache={},this}function _(S){let M={boundary:0,storage:0};return typeof S=="number"||typeof S=="boolean"?(M.boundary=4,M.storage=4):S.isVector2?(M.boundary=8,M.storage=8):S.isVector3||S.isColor?(M.boundary=16,M.storage=12):S.isVector4?(M.boundary=16,M.storage=16):S.isMatrix3?(M.boundary=48,M.storage=48):S.isMatrix4?(M.boundary=64,M.storage=64):S.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",S),M}function m(S){let M=S.target;M.removeEventListener("dispose",m);let x=o.indexOf(M.__bindingPointIndex);o.splice(x,1),n.deleteBuffer(s[M.id]),delete s[M.id],delete r[M.id]}function p(){for(let S in s)n.deleteBuffer(s[S]);o=[],s={},r={}}return{bind:c,update:l,dispose:p}}var kr=class{constructor(t={}){let{canvas:e=Ld(),context:i=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:u=!1,reverseDepthBuffer:d=!1}=t;this.isWebGLRenderer=!0;let f;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");f=i.getContextAttributes().alpha}else f=o;let g=new Uint32Array(4),_=new Int32Array(4),m=null,p=null,S=[],M=[];this.domElement=e,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this._outputColorSpace=xe,this.toneMapping=Yn,this.toneMappingExposure=1;let x=this,D=!1,T=0,R=0,C=null,b=-1,y=null,I=new ie,B=new ie,N=null,H=new kt(0),Y=0,V=e.width,rt=e.height,X=1,it=null,dt=null,vt=new ie(0,0,V,rt),zt=new ie(0,0,V,rt),Kt=!1,P=new Bs,k=!1,nt=!1,tt=new de,wt=new de,Dt=new L,Lt=new ie,Yt={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},Z=!1;function st(){return C===null?X:1}let A=i;function St(E,F){return e.getContext(E,F)}try{let E={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:u};if("setAttribute"in e&&e.setAttribute("data-engine",`three.js r${Ec}`),e.addEventListener("webglcontextlost",Q,!1),e.addEventListener("webglcontextrestored",_t,!1),e.addEventListener("webglcontextcreationerror",mt,!1),A===null){let F="webgl2";if(A=St(F,E),A===null)throw St(F)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(E){throw console.error("THREE.WebGLRenderer: "+E.message),E}let ot,Mt,ht,Ut,xt,w,v,z,$,et,J,Tt,ft,yt,Zt,at,Et,Nt,Bt,bt,$t,Xt,oe,U;function pt(){ot=new Xm(A),ot.init(),Xt=new E0(A,ot),Mt=new zm(A,ot,t,Xt),ht=new v0(A,ot),Mt.reverseDepthBuffer&&d&&ht.buffers.depth.setReversed(!0),Ut=new Zm(A),xt=new a0,w=new S0(A,ot,ht,xt,Mt,Xt,Ut),v=new km(x),z=new Wm(x),$=new tf(A),oe=new Om(A,$),et=new qm(A,$,Ut,oe),J=new Jm(A,et,$,Ut),Bt=new $m(A,Mt,w),at=new Hm(xt),Tt=new o0(x,v,z,ot,Mt,oe,at),ft=new R0(x,xt),yt=new l0,Zt=new m0(ot),Nt=new Fm(x,v,z,ht,J,f,c),Et=new x0(x,J,Mt),U=new C0(A,Ut,Mt,ht),bt=new Bm(A,ot,Ut),$t=new Ym(A,ot,Ut),Ut.programs=Tt.programs,x.capabilities=Mt,x.extensions=ot,x.properties=xt,x.renderLists=yt,x.shadowMap=Et,x.state=ht,x.info=Ut}pt();let q=new Qa(x,A);this.xr=q,this.getContext=function(){return A},this.getContextAttributes=function(){return A.getContextAttributes()},this.forceContextLoss=function(){let E=ot.get("WEBGL_lose_context");E&&E.loseContext()},this.forceContextRestore=function(){let E=ot.get("WEBGL_lose_context");E&&E.restoreContext()},this.getPixelRatio=function(){return X},this.setPixelRatio=function(E){E!==void 0&&(X=E,this.setSize(V,rt,!1))},this.getSize=function(E){return E.set(V,rt)},this.setSize=function(E,F,G=!0){if(q.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}V=E,rt=F,e.width=Math.floor(E*X),e.height=Math.floor(F*X),G===!0&&(e.style.width=E+"px",e.style.height=F+"px"),this.setViewport(0,0,E,F)},this.getDrawingBufferSize=function(E){return E.set(V*X,rt*X).floor()},this.setDrawingBufferSize=function(E,F,G){V=E,rt=F,X=G,e.width=Math.floor(E*G),e.height=Math.floor(F*G),this.setViewport(0,0,E,F)},this.getCurrentViewport=function(E){return E.copy(I)},this.getViewport=function(E){return E.copy(vt)},this.setViewport=function(E,F,G,W){E.isVector4?vt.set(E.x,E.y,E.z,E.w):vt.set(E,F,G,W),ht.viewport(I.copy(vt).multiplyScalar(X).round())},this.getScissor=function(E){return E.copy(zt)},this.setScissor=function(E,F,G,W){E.isVector4?zt.set(E.x,E.y,E.z,E.w):zt.set(E,F,G,W),ht.scissor(B.copy(zt).multiplyScalar(X).round())},this.getScissorTest=function(){return Kt},this.setScissorTest=function(E){ht.setScissorTest(Kt=E)},this.setOpaqueSort=function(E){it=E},this.setTransparentSort=function(E){dt=E},this.getClearColor=function(E){return E.copy(Nt.getClearColor())},this.setClearColor=function(){Nt.setClearColor.apply(Nt,arguments)},this.getClearAlpha=function(){return Nt.getClearAlpha()},this.setClearAlpha=function(){Nt.setClearAlpha.apply(Nt,arguments)},this.clear=function(E=!0,F=!0,G=!0){let W=0;if(E){let O=!1;if(C!==null){let ct=C.texture.format;O=ct===Pc||ct===Ic||ct===Cc}if(O){let ct=C.texture.type,gt=ct===Pn||ct===Mi||ct===Us||ct===es||ct===Ac||ct===Rc,At=Nt.getClearColor(),Rt=Nt.getClearAlpha(),Ht=At.r,Gt=At.g,Ct=At.b;gt?(g[0]=Ht,g[1]=Gt,g[2]=Ct,g[3]=Rt,A.clearBufferuiv(A.COLOR,0,g)):(_[0]=Ht,_[1]=Gt,_[2]=Ct,_[3]=Rt,A.clearBufferiv(A.COLOR,0,_))}else W|=A.COLOR_BUFFER_BIT}F&&(W|=A.DEPTH_BUFFER_BIT),G&&(W|=A.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),A.clear(W)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",Q,!1),e.removeEventListener("webglcontextrestored",_t,!1),e.removeEventListener("webglcontextcreationerror",mt,!1),yt.dispose(),Zt.dispose(),xt.dispose(),v.dispose(),z.dispose(),J.dispose(),oe.dispose(),U.dispose(),Tt.dispose(),q.dispose(),q.removeEventListener("sessionstart",jc),q.removeEventListener("sessionend",tl),ci.stop()};function Q(E){E.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),D=!0}function _t(){console.log("THREE.WebGLRenderer: Context Restored."),D=!1;let E=Ut.autoReset,F=Et.enabled,G=Et.autoUpdate,W=Et.needsUpdate,O=Et.type;pt(),Ut.autoReset=E,Et.enabled=F,Et.autoUpdate=G,Et.needsUpdate=W,Et.type=O}function mt(E){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",E.statusMessage)}function Vt(E){let F=E.target;F.removeEventListener("dispose",Vt),me(F)}function me(E){Re(E),xt.remove(E)}function Re(E){let F=xt.get(E).programs;F!==void 0&&(F.forEach(function(G){Tt.releaseProgram(G)}),E.isShaderMaterial&&Tt.releaseShaderCache(E))}this.renderBufferDirect=function(E,F,G,W,O,ct){F===null&&(F=Yt);let gt=O.isMesh&&O.matrixWorld.determinant()<0,At=Ru(E,F,G,W,O);ht.setMaterial(W,gt);let Rt=G.index,Ht=1;if(W.wireframe===!0){if(Rt=et.getWireframeAttribute(G),Rt===void 0)return;Ht=2}let Gt=G.drawRange,Ct=G.attributes.position,Qt=Gt.start*Ht,ae=(Gt.start+Gt.count)*Ht;ct!==null&&(Qt=Math.max(Qt,ct.start*Ht),ae=Math.min(ae,(ct.start+ct.count)*Ht)),Rt!==null?(Qt=Math.max(Qt,0),ae=Math.min(ae,Rt.count)):Ct!=null&&(Qt=Math.max(Qt,0),ae=Math.min(ae,Ct.count));let le=ae-Qt;if(le<0||le===1/0)return;oe.setup(O,W,At,G,Rt);let Oe,jt=bt;if(Rt!==null&&(Oe=$.get(Rt),jt=$t,jt.setIndex(Oe)),O.isMesh)W.wireframe===!0?(ht.setLineWidth(W.wireframeLinewidth*st()),jt.setMode(A.LINES)):jt.setMode(A.TRIANGLES);else if(O.isLine){let It=W.linewidth;It===void 0&&(It=1),ht.setLineWidth(It*st()),O.isLineSegments?jt.setMode(A.LINES):O.isLineLoop?jt.setMode(A.LINE_LOOP):jt.setMode(A.LINE_STRIP)}else O.isPoints?jt.setMode(A.POINTS):O.isSprite&&jt.setMode(A.TRIANGLES);if(O.isBatchedMesh)if(O._multiDrawInstances!==null)jt.renderMultiDrawInstances(O._multiDrawStarts,O._multiDrawCounts,O._multiDrawCount,O._multiDrawInstances);else if(ot.get("WEBGL_multi_draw"))jt.renderMultiDraw(O._multiDrawStarts,O._multiDrawCounts,O._multiDrawCount);else{let It=O._multiDrawStarts,vn=O._multiDrawCounts,te=O._multiDrawCount,je=Rt?$.get(Rt).bytesPerElement:1,Ii=xt.get(W).currentProgram.getUniforms();for(let ke=0;ke<te;ke++)Ii.setValue(A,"_gl_DrawID",ke),jt.render(It[ke]/je,vn[ke])}else if(O.isInstancedMesh)jt.renderInstances(Qt,le,O.count);else if(G.isInstancedBufferGeometry){let It=G._maxInstanceCount!==void 0?G._maxInstanceCount:1/0,vn=Math.min(G.instanceCount,It);jt.renderInstances(Qt,le,vn)}else jt.render(Qt,le)};function ee(E,F,G){E.transparent===!0&&E.side===ve&&E.forceSinglePass===!1?(E.side=Be,E.needsUpdate=!0,Js(E,F,G),E.side=$n,E.needsUpdate=!0,Js(E,F,G),E.side=ve):Js(E,F,G)}this.compile=function(E,F,G=null){G===null&&(G=E),p=Zt.get(G),p.init(F),M.push(p),G.traverseVisible(function(O){O.isLight&&O.layers.test(F.layers)&&(p.pushLight(O),O.castShadow&&p.pushShadow(O))}),E!==G&&E.traverseVisible(function(O){O.isLight&&O.layers.test(F.layers)&&(p.pushLight(O),O.castShadow&&p.pushShadow(O))}),p.setupLights();let W=new Set;return E.traverse(function(O){if(!(O.isMesh||O.isPoints||O.isLine||O.isSprite))return;let ct=O.material;if(ct)if(Array.isArray(ct))for(let gt=0;gt<ct.length;gt++){let At=ct[gt];ee(At,G,O),W.add(At)}else ee(ct,G,O),W.add(ct)}),M.pop(),p=null,W},this.compileAsync=function(E,F,G=null){let W=this.compile(E,F,G);return new Promise(O=>{function ct(){if(W.forEach(function(gt){xt.get(gt).currentProgram.isReady()&&W.delete(gt)}),W.size===0){O(E);return}setTimeout(ct,10)}ot.get("KHR_parallel_shader_compile")!==null?ct():setTimeout(ct,10)})};let Qe=null;function yn(E){Qe&&Qe(E)}function jc(){ci.stop()}function tl(){ci.start()}let ci=new Fh;ci.setAnimationLoop(yn),typeof self<"u"&&ci.setContext(self),this.setAnimationLoop=function(E){Qe=E,q.setAnimationLoop(E),E===null?ci.stop():ci.start()},q.addEventListener("sessionstart",jc),q.addEventListener("sessionend",tl),this.render=function(E,F){if(F!==void 0&&F.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(D===!0)return;if(E.matrixWorldAutoUpdate===!0&&E.updateMatrixWorld(),F.parent===null&&F.matrixWorldAutoUpdate===!0&&F.updateMatrixWorld(),q.enabled===!0&&q.isPresenting===!0&&(q.cameraAutoUpdate===!0&&q.updateCamera(F),F=q.getCamera()),E.isScene===!0&&E.onBeforeRender(x,E,F,C),p=Zt.get(E,M.length),p.init(F),M.push(p),wt.multiplyMatrices(F.projectionMatrix,F.matrixWorldInverse),P.setFromProjectionMatrix(wt),nt=this.localClippingEnabled,k=at.init(this.clippingPlanes,nt),m=yt.get(E,S.length),m.init(),S.push(m),q.enabled===!0&&q.isPresenting===!0){let ct=x.xr.getDepthSensingMesh();ct!==null&&xo(ct,F,-1/0,x.sortObjects)}xo(E,F,0,x.sortObjects),m.finish(),x.sortObjects===!0&&m.sort(it,dt),Z=q.enabled===!1||q.isPresenting===!1||q.hasDepthSensing()===!1,Z&&Nt.addToRenderList(m,E),this.info.render.frame++,k===!0&&at.beginShadows();let G=p.state.shadowsArray;Et.render(G,E,F),k===!0&&at.endShadows(),this.info.autoReset===!0&&this.info.reset();let W=m.opaque,O=m.transmissive;if(p.setupLights(),F.isArrayCamera){let ct=F.cameras;if(O.length>0)for(let gt=0,At=ct.length;gt<At;gt++){let Rt=ct[gt];nl(W,O,E,Rt)}Z&&Nt.render(E);for(let gt=0,At=ct.length;gt<At;gt++){let Rt=ct[gt];el(m,E,Rt,Rt.viewport)}}else O.length>0&&nl(W,O,E,F),Z&&Nt.render(E),el(m,E,F);C!==null&&(w.updateMultisampleRenderTarget(C),w.updateRenderTargetMipmap(C)),E.isScene===!0&&E.onAfterRender(x,E,F),oe.resetDefaultState(),b=-1,y=null,M.pop(),M.length>0?(p=M[M.length-1],k===!0&&at.setGlobalState(x.clippingPlanes,p.state.camera)):p=null,S.pop(),S.length>0?m=S[S.length-1]:m=null};function xo(E,F,G,W){if(E.visible===!1)return;if(E.layers.test(F.layers)){if(E.isGroup)G=E.renderOrder;else if(E.isLOD)E.autoUpdate===!0&&E.update(F);else if(E.isLight)p.pushLight(E),E.castShadow&&p.pushShadow(E);else if(E.isSprite){if(!E.frustumCulled||P.intersectsSprite(E)){W&&Lt.setFromMatrixPosition(E.matrixWorld).applyMatrix4(wt);let gt=J.update(E),At=E.material;At.visible&&m.push(E,gt,At,G,Lt.z,null)}}else if((E.isMesh||E.isLine||E.isPoints)&&(!E.frustumCulled||P.intersectsObject(E))){let gt=J.update(E),At=E.material;if(W&&(E.boundingSphere!==void 0?(E.boundingSphere===null&&E.computeBoundingSphere(),Lt.copy(E.boundingSphere.center)):(gt.boundingSphere===null&&gt.computeBoundingSphere(),Lt.copy(gt.boundingSphere.center)),Lt.applyMatrix4(E.matrixWorld).applyMatrix4(wt)),Array.isArray(At)){let Rt=gt.groups;for(let Ht=0,Gt=Rt.length;Ht<Gt;Ht++){let Ct=Rt[Ht],Qt=At[Ct.materialIndex];Qt&&Qt.visible&&m.push(E,gt,Qt,G,Lt.z,Ct)}}else At.visible&&m.push(E,gt,At,G,Lt.z,null)}}let ct=E.children;for(let gt=0,At=ct.length;gt<At;gt++)xo(ct[gt],F,G,W)}function el(E,F,G,W){let O=E.opaque,ct=E.transmissive,gt=E.transparent;p.setupLightsView(G),k===!0&&at.setGlobalState(x.clippingPlanes,G),W&&ht.viewport(I.copy(W)),O.length>0&&$s(O,F,G),ct.length>0&&$s(ct,F,G),gt.length>0&&$s(gt,F,G),ht.buffers.depth.setTest(!0),ht.buffers.depth.setMask(!0),ht.buffers.color.setMask(!0),ht.setPolygonOffset(!1)}function nl(E,F,G,W){if((G.isScene===!0?G.overrideMaterial:null)!==null)return;p.state.transmissionRenderTarget[W.id]===void 0&&(p.state.transmissionRenderTarget[W.id]=new Ln(1,1,{generateMipmaps:!0,type:ot.has("EXT_color_buffer_half_float")||ot.has("EXT_color_buffer_float")?Gs:Pn,minFilter:vi,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:Jt.workingColorSpace}));let ct=p.state.transmissionRenderTarget[W.id],gt=W.viewport||I;ct.setSize(gt.z,gt.w);let At=x.getRenderTarget();x.setRenderTarget(ct),x.getClearColor(H),Y=x.getClearAlpha(),Y<1&&x.setClearColor(16777215,.5),x.clear(),Z&&Nt.render(G);let Rt=x.toneMapping;x.toneMapping=Yn;let Ht=W.viewport;if(W.viewport!==void 0&&(W.viewport=void 0),p.setupLightsView(W),k===!0&&at.setGlobalState(x.clippingPlanes,W),$s(E,G,W),w.updateMultisampleRenderTarget(ct),w.updateRenderTargetMipmap(ct),ot.has("WEBGL_multisampled_render_to_texture")===!1){let Gt=!1;for(let Ct=0,Qt=F.length;Ct<Qt;Ct++){let ae=F[Ct],le=ae.object,Oe=ae.geometry,jt=ae.material,It=ae.group;if(jt.side===ve&&le.layers.test(W.layers)){let vn=jt.side;jt.side=Be,jt.needsUpdate=!0,il(le,G,W,Oe,jt,It),jt.side=vn,jt.needsUpdate=!0,Gt=!0}}Gt===!0&&(w.updateMultisampleRenderTarget(ct),w.updateRenderTargetMipmap(ct))}x.setRenderTarget(At),x.setClearColor(H,Y),Ht!==void 0&&(W.viewport=Ht),x.toneMapping=Rt}function $s(E,F,G){let W=F.isScene===!0?F.overrideMaterial:null;for(let O=0,ct=E.length;O<ct;O++){let gt=E[O],At=gt.object,Rt=gt.geometry,Ht=W===null?gt.material:W,Gt=gt.group;At.layers.test(G.layers)&&il(At,F,G,Rt,Ht,Gt)}}function il(E,F,G,W,O,ct){E.onBeforeRender(x,F,G,W,O,ct),E.modelViewMatrix.multiplyMatrices(G.matrixWorldInverse,E.matrixWorld),E.normalMatrix.getNormalMatrix(E.modelViewMatrix),O.onBeforeRender(x,F,G,W,E,ct),O.transparent===!0&&O.side===ve&&O.forceSinglePass===!1?(O.side=Be,O.needsUpdate=!0,x.renderBufferDirect(G,F,W,O,E,ct),O.side=$n,O.needsUpdate=!0,x.renderBufferDirect(G,F,W,O,E,ct),O.side=ve):x.renderBufferDirect(G,F,W,O,E,ct),E.onAfterRender(x,F,G,W,O,ct)}function Js(E,F,G){F.isScene!==!0&&(F=Yt);let W=xt.get(E),O=p.state.lights,ct=p.state.shadowsArray,gt=O.state.version,At=Tt.getParameters(E,O.state,ct,F,G),Rt=Tt.getProgramCacheKey(At),Ht=W.programs;W.environment=E.isMeshStandardMaterial?F.environment:null,W.fog=F.fog,W.envMap=(E.isMeshStandardMaterial?z:v).get(E.envMap||W.environment),W.envMapRotation=W.environment!==null&&E.envMap===null?F.environmentRotation:E.envMapRotation,Ht===void 0&&(E.addEventListener("dispose",Vt),Ht=new Map,W.programs=Ht);let Gt=Ht.get(Rt);if(Gt!==void 0){if(W.currentProgram===Gt&&W.lightsStateVersion===gt)return rl(E,At),Gt}else At.uniforms=Tt.getUniforms(E),E.onBeforeCompile(At,x),Gt=Tt.acquireProgram(At,Rt),Ht.set(Rt,Gt),W.uniforms=At.uniforms;let Ct=W.uniforms;return(!E.isShaderMaterial&&!E.isRawShaderMaterial||E.clipping===!0)&&(Ct.clippingPlanes=at.uniform),rl(E,At),W.needsLights=Iu(E),W.lightsStateVersion=gt,W.needsLights&&(Ct.ambientLightColor.value=O.state.ambient,Ct.lightProbe.value=O.state.probe,Ct.directionalLights.value=O.state.directional,Ct.directionalLightShadows.value=O.state.directionalShadow,Ct.spotLights.value=O.state.spot,Ct.spotLightShadows.value=O.state.spotShadow,Ct.rectAreaLights.value=O.state.rectArea,Ct.ltc_1.value=O.state.rectAreaLTC1,Ct.ltc_2.value=O.state.rectAreaLTC2,Ct.pointLights.value=O.state.point,Ct.pointLightShadows.value=O.state.pointShadow,Ct.hemisphereLights.value=O.state.hemi,Ct.directionalShadowMap.value=O.state.directionalShadowMap,Ct.directionalShadowMatrix.value=O.state.directionalShadowMatrix,Ct.spotShadowMap.value=O.state.spotShadowMap,Ct.spotLightMatrix.value=O.state.spotLightMatrix,Ct.spotLightMap.value=O.state.spotLightMap,Ct.pointShadowMap.value=O.state.pointShadowMap,Ct.pointShadowMatrix.value=O.state.pointShadowMatrix),W.currentProgram=Gt,W.uniformsList=null,Gt}function sl(E){if(E.uniformsList===null){let F=E.currentProgram.getUniforms();E.uniformsList=Ki.seqWithValue(F.seq,E.uniforms)}return E.uniformsList}function rl(E,F){let G=xt.get(E);G.outputColorSpace=F.outputColorSpace,G.batching=F.batching,G.batchingColor=F.batchingColor,G.instancing=F.instancing,G.instancingColor=F.instancingColor,G.instancingMorph=F.instancingMorph,G.skinning=F.skinning,G.morphTargets=F.morphTargets,G.morphNormals=F.morphNormals,G.morphColors=F.morphColors,G.morphTargetsCount=F.morphTargetsCount,G.numClippingPlanes=F.numClippingPlanes,G.numIntersection=F.numClipIntersection,G.vertexAlphas=F.vertexAlphas,G.vertexTangents=F.vertexTangents,G.toneMapping=F.toneMapping}function Ru(E,F,G,W,O){F.isScene!==!0&&(F=Yt),w.resetTextureUnits();let ct=F.fog,gt=W.isMeshStandardMaterial?F.environment:null,At=C===null?x.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:cs,Rt=(W.isMeshStandardMaterial?z:v).get(W.envMap||gt),Ht=W.vertexColors===!0&&!!G.attributes.color&&G.attributes.color.itemSize===4,Gt=!!G.attributes.tangent&&(!!W.normalMap||W.anisotropy>0),Ct=!!G.morphAttributes.position,Qt=!!G.morphAttributes.normal,ae=!!G.morphAttributes.color,le=Yn;W.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(le=x.toneMapping);let Oe=G.morphAttributes.position||G.morphAttributes.normal||G.morphAttributes.color,jt=Oe!==void 0?Oe.length:0,It=xt.get(W),vn=p.state.lights;if(k===!0&&(nt===!0||E!==y)){let Ye=E===y&&W.id===b;at.setState(W,E,Ye)}let te=!1;W.version===It.__version?(It.needsLights&&It.lightsStateVersion!==vn.state.version||It.outputColorSpace!==At||O.isBatchedMesh&&It.batching===!1||!O.isBatchedMesh&&It.batching===!0||O.isBatchedMesh&&It.batchingColor===!0&&O.colorTexture===null||O.isBatchedMesh&&It.batchingColor===!1&&O.colorTexture!==null||O.isInstancedMesh&&It.instancing===!1||!O.isInstancedMesh&&It.instancing===!0||O.isSkinnedMesh&&It.skinning===!1||!O.isSkinnedMesh&&It.skinning===!0||O.isInstancedMesh&&It.instancingColor===!0&&O.instanceColor===null||O.isInstancedMesh&&It.instancingColor===!1&&O.instanceColor!==null||O.isInstancedMesh&&It.instancingMorph===!0&&O.morphTexture===null||O.isInstancedMesh&&It.instancingMorph===!1&&O.morphTexture!==null||It.envMap!==Rt||W.fog===!0&&It.fog!==ct||It.numClippingPlanes!==void 0&&(It.numClippingPlanes!==at.numPlanes||It.numIntersection!==at.numIntersection)||It.vertexAlphas!==Ht||It.vertexTangents!==Gt||It.morphTargets!==Ct||It.morphNormals!==Qt||It.morphColors!==ae||It.toneMapping!==le||It.morphTargetsCount!==jt)&&(te=!0):(te=!0,It.__version=W.version);let je=It.currentProgram;te===!0&&(je=Js(W,F,O));let Ii=!1,ke=!1,ys=!1,he=je.getUniforms(),cn=It.uniforms;if(ht.useProgram(je.program)&&(Ii=!0,ke=!0,ys=!0),W.id!==b&&(b=W.id,ke=!0),Ii||y!==E){ht.buffers.depth.getReversed()?(tt.copy(E.projectionMatrix),Ud(tt),Nd(tt),he.setValue(A,"projectionMatrix",tt)):he.setValue(A,"projectionMatrix",E.projectionMatrix),he.setValue(A,"viewMatrix",E.matrixWorldInverse);let On=he.map.cameraPosition;On!==void 0&&On.setValue(A,Dt.setFromMatrixPosition(E.matrixWorld)),Mt.logarithmicDepthBuffer&&he.setValue(A,"logDepthBufFC",2/(Math.log(E.far+1)/Math.LN2)),(W.isMeshPhongMaterial||W.isMeshToonMaterial||W.isMeshLambertMaterial||W.isMeshBasicMaterial||W.isMeshStandardMaterial||W.isShaderMaterial)&&he.setValue(A,"isOrthographic",E.isOrthographicCamera===!0),y!==E&&(y=E,ke=!0,ys=!0)}if(O.isSkinnedMesh){he.setOptional(A,O,"bindMatrix"),he.setOptional(A,O,"bindMatrixInverse");let Ye=O.skeleton;Ye&&(Ye.boneTexture===null&&Ye.computeBoneTexture(),he.setValue(A,"boneTexture",Ye.boneTexture,w))}O.isBatchedMesh&&(he.setOptional(A,O,"batchingTexture"),he.setValue(A,"batchingTexture",O._matricesTexture,w),he.setOptional(A,O,"batchingIdTexture"),he.setValue(A,"batchingIdTexture",O._indirectTexture,w),he.setOptional(A,O,"batchingColorTexture"),O._colorsTexture!==null&&he.setValue(A,"batchingColorTexture",O._colorsTexture,w));let vs=G.morphAttributes;if((vs.position!==void 0||vs.normal!==void 0||vs.color!==void 0)&&Bt.update(O,G,je),(ke||It.receiveShadow!==O.receiveShadow)&&(It.receiveShadow=O.receiveShadow,he.setValue(A,"receiveShadow",O.receiveShadow)),W.isMeshGouraudMaterial&&W.envMap!==null&&(cn.envMap.value=Rt,cn.flipEnvMap.value=Rt.isCubeTexture&&Rt.isRenderTargetTexture===!1?-1:1),W.isMeshStandardMaterial&&W.envMap===null&&F.environment!==null&&(cn.envMapIntensity.value=F.environmentIntensity),ke&&(he.setValue(A,"toneMappingExposure",x.toneMappingExposure),It.needsLights&&Cu(cn,ys),ct&&W.fog===!0&&ft.refreshFogUniforms(cn,ct),ft.refreshMaterialUniforms(cn,W,X,rt,p.state.transmissionRenderTarget[E.id]),Ki.upload(A,sl(It),cn,w)),W.isShaderMaterial&&W.uniformsNeedUpdate===!0&&(Ki.upload(A,sl(It),cn,w),W.uniformsNeedUpdate=!1),W.isSpriteMaterial&&he.setValue(A,"center",O.center),he.setValue(A,"modelViewMatrix",O.modelViewMatrix),he.setValue(A,"normalMatrix",O.normalMatrix),he.setValue(A,"modelMatrix",O.matrixWorld),W.isShaderMaterial||W.isRawShaderMaterial){let Ye=W.uniformsGroups;for(let On=0,Bn=Ye.length;On<Bn;On++){let ol=Ye[On];U.update(ol,je),U.bind(ol,je)}}return je}function Cu(E,F){E.ambientLightColor.needsUpdate=F,E.lightProbe.needsUpdate=F,E.directionalLights.needsUpdate=F,E.directionalLightShadows.needsUpdate=F,E.pointLights.needsUpdate=F,E.pointLightShadows.needsUpdate=F,E.spotLights.needsUpdate=F,E.spotLightShadows.needsUpdate=F,E.rectAreaLights.needsUpdate=F,E.hemisphereLights.needsUpdate=F}function Iu(E){return E.isMeshLambertMaterial||E.isMeshToonMaterial||E.isMeshPhongMaterial||E.isMeshStandardMaterial||E.isShadowMaterial||E.isShaderMaterial&&E.lights===!0}this.getActiveCubeFace=function(){return T},this.getActiveMipmapLevel=function(){return R},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(E,F,G){xt.get(E.texture).__webglTexture=F,xt.get(E.depthTexture).__webglTexture=G;let W=xt.get(E);W.__hasExternalTextures=!0,W.__autoAllocateDepthBuffer=G===void 0,W.__autoAllocateDepthBuffer||ot.has("WEBGL_multisampled_render_to_texture")===!0&&(console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"),W.__useRenderToTexture=!1)},this.setRenderTargetFramebuffer=function(E,F){let G=xt.get(E);G.__webglFramebuffer=F,G.__useDefaultFramebuffer=F===void 0},this.setRenderTarget=function(E,F=0,G=0){C=E,T=F,R=G;let W=!0,O=null,ct=!1,gt=!1;if(E){let Rt=xt.get(E);if(Rt.__useDefaultFramebuffer!==void 0)ht.bindFramebuffer(A.FRAMEBUFFER,null),W=!1;else if(Rt.__webglFramebuffer===void 0)w.setupRenderTarget(E);else if(Rt.__hasExternalTextures)w.rebindTextures(E,xt.get(E.texture).__webglTexture,xt.get(E.depthTexture).__webglTexture);else if(E.depthBuffer){let Ct=E.depthTexture;if(Rt.__boundDepthTexture!==Ct){if(Ct!==null&&xt.has(Ct)&&(E.width!==Ct.image.width||E.height!==Ct.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");w.setupDepthRenderbuffer(E)}}let Ht=E.texture;(Ht.isData3DTexture||Ht.isDataArrayTexture||Ht.isCompressedArrayTexture)&&(gt=!0);let Gt=xt.get(E).__webglFramebuffer;E.isWebGLCubeRenderTarget?(Array.isArray(Gt[F])?O=Gt[F][G]:O=Gt[F],ct=!0):E.samples>0&&w.useMultisampledRTT(E)===!1?O=xt.get(E).__webglMultisampledFramebuffer:Array.isArray(Gt)?O=Gt[G]:O=Gt,I.copy(E.viewport),B.copy(E.scissor),N=E.scissorTest}else I.copy(vt).multiplyScalar(X).floor(),B.copy(zt).multiplyScalar(X).floor(),N=Kt;if(ht.bindFramebuffer(A.FRAMEBUFFER,O)&&W&&ht.drawBuffers(E,O),ht.viewport(I),ht.scissor(B),ht.setScissorTest(N),ct){let Rt=xt.get(E.texture);A.framebufferTexture2D(A.FRAMEBUFFER,A.COLOR_ATTACHMENT0,A.TEXTURE_CUBE_MAP_POSITIVE_X+F,Rt.__webglTexture,G)}else if(gt){let Rt=xt.get(E.texture),Ht=F||0;A.framebufferTextureLayer(A.FRAMEBUFFER,A.COLOR_ATTACHMENT0,Rt.__webglTexture,G||0,Ht)}b=-1},this.readRenderTargetPixels=function(E,F,G,W,O,ct,gt){if(!(E&&E.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let At=xt.get(E).__webglFramebuffer;if(E.isWebGLCubeRenderTarget&&gt!==void 0&&(At=At[gt]),At){ht.bindFramebuffer(A.FRAMEBUFFER,At);try{let Rt=E.texture,Ht=Rt.format,Gt=Rt.type;if(!Mt.textureFormatReadable(Ht)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!Mt.textureTypeReadable(Gt)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}F>=0&&F<=E.width-W&&G>=0&&G<=E.height-O&&A.readPixels(F,G,W,O,Xt.convert(Ht),Xt.convert(Gt),ct)}finally{let Rt=C!==null?xt.get(C).__webglFramebuffer:null;ht.bindFramebuffer(A.FRAMEBUFFER,Rt)}}},this.readRenderTargetPixelsAsync=async function(E,F,G,W,O,ct,gt){if(!(E&&E.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let At=xt.get(E).__webglFramebuffer;if(E.isWebGLCubeRenderTarget&&gt!==void 0&&(At=At[gt]),At){let Rt=E.texture,Ht=Rt.format,Gt=Rt.type;if(!Mt.textureFormatReadable(Ht))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!Mt.textureTypeReadable(Gt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");if(F>=0&&F<=E.width-W&&G>=0&&G<=E.height-O){ht.bindFramebuffer(A.FRAMEBUFFER,At);let Ct=A.createBuffer();A.bindBuffer(A.PIXEL_PACK_BUFFER,Ct),A.bufferData(A.PIXEL_PACK_BUFFER,ct.byteLength,A.STREAM_READ),A.readPixels(F,G,W,O,Xt.convert(Ht),Xt.convert(Gt),0);let Qt=C!==null?xt.get(C).__webglFramebuffer:null;ht.bindFramebuffer(A.FRAMEBUFFER,Qt);let ae=A.fenceSync(A.SYNC_GPU_COMMANDS_COMPLETE,0);return A.flush(),await Dd(A,ae,4),A.bindBuffer(A.PIXEL_PACK_BUFFER,Ct),A.getBufferSubData(A.PIXEL_PACK_BUFFER,0,ct),A.deleteBuffer(Ct),A.deleteSync(ae),ct}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")}},this.copyFramebufferToTexture=function(E,F=null,G=0){E.isTexture!==!0&&(Ts("WebGLRenderer: copyFramebufferToTexture function signature has changed."),F=arguments[0]||null,E=arguments[1]);let W=Math.pow(2,-G),O=Math.floor(E.image.width*W),ct=Math.floor(E.image.height*W),gt=F!==null?F.x:0,At=F!==null?F.y:0;w.setTexture2D(E,0),A.copyTexSubImage2D(A.TEXTURE_2D,G,0,0,gt,At,O,ct),ht.unbindTexture()},this.copyTextureToTexture=function(E,F,G=null,W=null,O=0){E.isTexture!==!0&&(Ts("WebGLRenderer: copyTextureToTexture function signature has changed."),W=arguments[0]||null,E=arguments[1],F=arguments[2],O=arguments[3]||0,G=null);let ct,gt,At,Rt,Ht,Gt,Ct,Qt,ae,le=E.isCompressedTexture?E.mipmaps[O]:E.image;G!==null?(ct=G.max.x-G.min.x,gt=G.max.y-G.min.y,At=G.isBox3?G.max.z-G.min.z:1,Rt=G.min.x,Ht=G.min.y,Gt=G.isBox3?G.min.z:0):(ct=le.width,gt=le.height,At=le.depth||1,Rt=0,Ht=0,Gt=0),W!==null?(Ct=W.x,Qt=W.y,ae=W.z):(Ct=0,Qt=0,ae=0);let Oe=Xt.convert(F.format),jt=Xt.convert(F.type),It;F.isData3DTexture?(w.setTexture3D(F,0),It=A.TEXTURE_3D):F.isDataArrayTexture||F.isCompressedArrayTexture?(w.setTexture2DArray(F,0),It=A.TEXTURE_2D_ARRAY):(w.setTexture2D(F,0),It=A.TEXTURE_2D),A.pixelStorei(A.UNPACK_FLIP_Y_WEBGL,F.flipY),A.pixelStorei(A.UNPACK_PREMULTIPLY_ALPHA_WEBGL,F.premultiplyAlpha),A.pixelStorei(A.UNPACK_ALIGNMENT,F.unpackAlignment);let vn=A.getParameter(A.UNPACK_ROW_LENGTH),te=A.getParameter(A.UNPACK_IMAGE_HEIGHT),je=A.getParameter(A.UNPACK_SKIP_PIXELS),Ii=A.getParameter(A.UNPACK_SKIP_ROWS),ke=A.getParameter(A.UNPACK_SKIP_IMAGES);A.pixelStorei(A.UNPACK_ROW_LENGTH,le.width),A.pixelStorei(A.UNPACK_IMAGE_HEIGHT,le.height),A.pixelStorei(A.UNPACK_SKIP_PIXELS,Rt),A.pixelStorei(A.UNPACK_SKIP_ROWS,Ht),A.pixelStorei(A.UNPACK_SKIP_IMAGES,Gt);let ys=E.isDataArrayTexture||E.isData3DTexture,he=F.isDataArrayTexture||F.isData3DTexture;if(E.isRenderTargetTexture||E.isDepthTexture){let cn=xt.get(E),vs=xt.get(F),Ye=xt.get(cn.__renderTarget),On=xt.get(vs.__renderTarget);ht.bindFramebuffer(A.READ_FRAMEBUFFER,Ye.__webglFramebuffer),ht.bindFramebuffer(A.DRAW_FRAMEBUFFER,On.__webglFramebuffer);for(let Bn=0;Bn<At;Bn++)ys&&A.framebufferTextureLayer(A.READ_FRAMEBUFFER,A.COLOR_ATTACHMENT0,xt.get(E).__webglTexture,O,Gt+Bn),E.isDepthTexture?(he&&A.framebufferTextureLayer(A.DRAW_FRAMEBUFFER,A.COLOR_ATTACHMENT0,xt.get(F).__webglTexture,O,ae+Bn),A.blitFramebuffer(Rt,Ht,ct,gt,Ct,Qt,ct,gt,A.DEPTH_BUFFER_BIT,A.NEAREST)):he?A.copyTexSubImage3D(It,O,Ct,Qt,ae+Bn,Rt,Ht,ct,gt):A.copyTexSubImage2D(It,O,Ct,Qt,ae+Bn,Rt,Ht,ct,gt);ht.bindFramebuffer(A.READ_FRAMEBUFFER,null),ht.bindFramebuffer(A.DRAW_FRAMEBUFFER,null)}else he?E.isDataTexture||E.isData3DTexture?A.texSubImage3D(It,O,Ct,Qt,ae,ct,gt,At,Oe,jt,le.data):F.isCompressedArrayTexture?A.compressedTexSubImage3D(It,O,Ct,Qt,ae,ct,gt,At,Oe,le.data):A.texSubImage3D(It,O,Ct,Qt,ae,ct,gt,At,Oe,jt,le):E.isDataTexture?A.texSubImage2D(A.TEXTURE_2D,O,Ct,Qt,ct,gt,Oe,jt,le.data):E.isCompressedTexture?A.compressedTexSubImage2D(A.TEXTURE_2D,O,Ct,Qt,le.width,le.height,Oe,le.data):A.texSubImage2D(A.TEXTURE_2D,O,Ct,Qt,ct,gt,Oe,jt,le);A.pixelStorei(A.UNPACK_ROW_LENGTH,vn),A.pixelStorei(A.UNPACK_IMAGE_HEIGHT,te),A.pixelStorei(A.UNPACK_SKIP_PIXELS,je),A.pixelStorei(A.UNPACK_SKIP_ROWS,Ii),A.pixelStorei(A.UNPACK_SKIP_IMAGES,ke),O===0&&F.generateMipmaps&&A.generateMipmap(It),ht.unbindTexture()},this.copyTextureToTexture3D=function(E,F,G=null,W=null,O=0){return E.isTexture!==!0&&(Ts("WebGLRenderer: copyTextureToTexture3D function signature has changed."),G=arguments[0]||null,W=arguments[1]||null,E=arguments[2],F=arguments[3],O=arguments[4]||0),Ts('WebGLRenderer: copyTextureToTexture3D function has been deprecated. Use "copyTextureToTexture" instead.'),this.copyTextureToTexture(E,F,G,W,O)},this.initRenderTarget=function(E){xt.get(E).__webglFramebuffer===void 0&&w.setupRenderTarget(E)},this.initTexture=function(E){E.isCubeTexture?w.setTextureCube(E,0):E.isData3DTexture?w.setTexture3D(E,0):E.isDataArrayTexture||E.isCompressedArrayTexture?w.setTexture2DArray(E,0):w.setTexture2D(E,0),ht.unbindTexture()},this.resetState=function(){T=0,R=0,C=null,ht.reset(),oe.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Cn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t;let e=this.getContext();e.drawingBufferColorspace=Jt._getDrawingBufferColorSpace(t),e.unpackColorSpace=Jt._getUnpackColorSpace()}};var Vr=class n{constructor(t,e=1,i=1e3){this.isFog=!0,this.name="",this.color=new kt(t),this.near=e,this.far=i}clone(){return new n(this.color,this.near,this.far)}toJSON(){return{type:"Fog",name:this.name,color:this.color.getHex(),near:this.near,far:this.far}}},Gr=class extends Ue{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new un,this.environmentIntensity=1,this.environmentRotation=new un,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),t.background!==null&&(this.background=t.background.clone()),t.environment!==null&&(this.environment=t.environment.clone()),t.fog!==null&&(this.fog=t.fog.clone()),this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,this.backgroundRotation.copy(t.backgroundRotation),this.environmentIntensity=t.environmentIntensity,this.environmentRotation.copy(t.environmentRotation),t.overrideMaterial!==null&&(this.overrideMaterial=t.overrideMaterial.clone()),this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){let e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(e.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(e.object.backgroundIntensity=this.backgroundIntensity),e.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(e.object.environmentIntensity=this.environmentIntensity),e.object.environmentRotation=this.environmentRotation.toArray(),e}};var Qn=class extends We{constructor(t,e,i,s,r,o,a,c,l){super(t,e,i,s,r,o,a,c,l),this.isCanvasTexture=!0,this.needsUpdate=!0}},$e=class{constructor(){this.type="Curve",this.arcLengthDivisions=200}getPoint(){return console.warn("THREE.Curve: .getPoint() not implemented."),null}getPointAt(t,e){let i=this.getUtoTmapping(t);return this.getPoint(i,e)}getPoints(t=5){let e=[];for(let i=0;i<=t;i++)e.push(this.getPoint(i/t));return e}getSpacedPoints(t=5){let e=[];for(let i=0;i<=t;i++)e.push(this.getPointAt(i/t));return e}getLength(){let t=this.getLengths();return t[t.length-1]}getLengths(t=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===t+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;let e=[],i,s=this.getPoint(0),r=0;e.push(0);for(let o=1;o<=t;o++)i=this.getPoint(o/t),r+=i.distanceTo(s),e.push(r),s=i;return this.cacheArcLengths=e,e}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(t,e){let i=this.getLengths(),s=0,r=i.length,o;e?o=e:o=t*i[r-1];let a=0,c=r-1,l;for(;a<=c;)if(s=Math.floor(a+(c-a)/2),l=i[s]-o,l<0)a=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,i[s]===o)return s/(r-1);let h=i[s],d=i[s+1]-h,f=(o-h)/d;return(s+f)/(r-1)}getTangent(t,e){let s=t-1e-4,r=t+1e-4;s<0&&(s=0),r>1&&(r=1);let o=this.getPoint(s),a=this.getPoint(r),c=e||(o.isVector2?new j:new L);return c.copy(a).sub(o).normalize(),c}getTangentAt(t,e){let i=this.getUtoTmapping(t);return this.getTangent(i,e)}computeFrenetFrames(t,e){let i=new L,s=[],r=[],o=[],a=new L,c=new de;for(let f=0;f<=t;f++){let g=f/t;s[f]=this.getTangentAt(g,new L)}r[0]=new L,o[0]=new L;let l=Number.MAX_VALUE,h=Math.abs(s[0].x),u=Math.abs(s[0].y),d=Math.abs(s[0].z);h<=l&&(l=h,i.set(1,0,0)),u<=l&&(l=u,i.set(0,1,0)),d<=l&&i.set(0,0,1),a.crossVectors(s[0],i).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let f=1;f<=t;f++){if(r[f]=r[f-1].clone(),o[f]=o[f-1].clone(),a.crossVectors(s[f-1],s[f]),a.length()>Number.EPSILON){a.normalize();let g=Math.acos(ye(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(c.makeRotationAxis(a,g))}o[f].crossVectors(s[f],r[f])}if(e===!0){let f=Math.acos(ye(r[0].dot(r[t]),-1,1));f/=t,s[0].dot(a.crossVectors(r[0],r[t]))>0&&(f=-f);for(let g=1;g<=t;g++)r[g].applyMatrix4(c.makeRotationAxis(s[g],f*g)),o[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}toJSON(){let t={metadata:{version:4.6,type:"Curve",generator:"Curve.toJSON"}};return t.arcLengthDivisions=this.arcLengthDivisions,t.type=this.type,t}fromJSON(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}},zs=class extends $e{constructor(t=0,e=0,i=1,s=1,r=0,o=Math.PI*2,a=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=t,this.aY=e,this.xRadius=i,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=o,this.aClockwise=a,this.aRotation=c}getPoint(t,e=new j){let i=e,s=Math.PI*2,r=this.aEndAngle-this.aStartAngle,o=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(o?r=0:r=s),this.aClockwise===!0&&!o&&(r===s?r=-s:r=r-s);let a=this.aStartAngle+t*r,c=this.aX+this.xRadius*Math.cos(a),l=this.aY+this.yRadius*Math.sin(a);if(this.aRotation!==0){let h=Math.cos(this.aRotation),u=Math.sin(this.aRotation),d=c-this.aX,f=l-this.aY;c=d*h-f*u+this.aX,l=d*u+f*h+this.aY}return i.set(c,l)}copy(t){return super.copy(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}toJSON(){let t=super.toJSON();return t.aX=this.aX,t.aY=this.aY,t.xRadius=this.xRadius,t.yRadius=this.yRadius,t.aStartAngle=this.aStartAngle,t.aEndAngle=this.aEndAngle,t.aClockwise=this.aClockwise,t.aRotation=this.aRotation,t}fromJSON(t){return super.fromJSON(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}},ja=class extends zs{constructor(t,e,i,s,r,o){super(t,e,i,i,s,r,o),this.isArcCurve=!0,this.type="ArcCurve"}};function Uc(){let n=0,t=0,e=0,i=0;function s(r,o,a,c){n=r,t=a,e=-3*r+3*o-2*a-c,i=2*r-2*o+a+c}return{initCatmullRom:function(r,o,a,c,l){s(o,a,l*(a-r),l*(c-o))},initNonuniformCatmullRom:function(r,o,a,c,l,h,u){let d=(o-r)/l-(a-r)/(l+h)+(a-o)/h,f=(a-o)/h-(c-o)/(h+u)+(c-a)/u;d*=h,f*=h,s(o,a,d,f)},calc:function(r){let o=r*r,a=o*r;return n+t*r+e*o+i*a}}}var xr=new L,Yo=new Uc,Zo=new Uc,$o=new Uc,tc=class extends $e{constructor(t=[],e=!1,i="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=t,this.closed=e,this.curveType=i,this.tension=s}getPoint(t,e=new L){let i=e,s=this.points,r=s.length,o=(r-(this.closed?0:1))*t,a=Math.floor(o),c=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:c===0&&a===r-1&&(a=r-2,c=1);let l,h;this.closed||a>0?l=s[(a-1)%r]:(xr.subVectors(s[0],s[1]).add(s[0]),l=xr);let u=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?h=s[(a+2)%r]:(xr.subVectors(s[r-1],s[r-2]).add(s[r-1]),h=xr),this.curveType==="centripetal"||this.curveType==="chordal"){let f=this.curveType==="chordal"?.5:.25,g=Math.pow(l.distanceToSquared(u),f),_=Math.pow(u.distanceToSquared(d),f),m=Math.pow(d.distanceToSquared(h),f);_<1e-4&&(_=1),g<1e-4&&(g=_),m<1e-4&&(m=_),Yo.initNonuniformCatmullRom(l.x,u.x,d.x,h.x,g,_,m),Zo.initNonuniformCatmullRom(l.y,u.y,d.y,h.y,g,_,m),$o.initNonuniformCatmullRom(l.z,u.z,d.z,h.z,g,_,m)}else this.curveType==="catmullrom"&&(Yo.initCatmullRom(l.x,u.x,d.x,h.x,this.tension),Zo.initCatmullRom(l.y,u.y,d.y,h.y,this.tension),$o.initCatmullRom(l.z,u.z,d.z,h.z,this.tension));return i.set(Yo.calc(c),Zo.calc(c),$o.calc(c)),i}copy(t){super.copy(t),this.points=[];for(let e=0,i=t.points.length;e<i;e++){let s=t.points[e];this.points.push(s.clone())}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}toJSON(){let t=super.toJSON();t.points=[];for(let e=0,i=this.points.length;e<i;e++){let s=this.points[e];t.points.push(s.toArray())}return t.closed=this.closed,t.curveType=this.curveType,t.tension=this.tension,t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,i=t.points.length;e<i;e++){let s=t.points[e];this.points.push(new L().fromArray(s))}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}};function ch(n,t,e,i,s){let r=(i-t)*.5,o=(s-e)*.5,a=n*n,c=n*a;return(2*e-2*i+r+o)*c+(-3*e+3*i-2*r-o)*a+r*n+e}function I0(n,t){let e=1-n;return e*e*t}function P0(n,t){return 2*(1-n)*n*t}function L0(n,t){return n*n*t}function Ps(n,t,e,i){return I0(n,t)+P0(n,e)+L0(n,i)}function D0(n,t){let e=1-n;return e*e*e*t}function U0(n,t){let e=1-n;return 3*e*e*n*t}function N0(n,t){return 3*(1-n)*n*n*t}function F0(n,t){return n*n*n*t}function Ls(n,t,e,i,s){return D0(n,t)+U0(n,e)+N0(n,i)+F0(n,s)}var Wr=class extends $e{constructor(t=new j,e=new j,i=new j,s=new j){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=t,this.v1=e,this.v2=i,this.v3=s}getPoint(t,e=new j){let i=e,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(Ls(t,s.x,r.x,o.x,a.x),Ls(t,s.y,r.y,o.y,a.y)),i}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}},ec=class extends $e{constructor(t=new L,e=new L,i=new L,s=new L){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=t,this.v1=e,this.v2=i,this.v3=s}getPoint(t,e=new L){let i=e,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(Ls(t,s.x,r.x,o.x,a.x),Ls(t,s.y,r.y,o.y,a.y),Ls(t,s.z,r.z,o.z,a.z)),i}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}},Xr=class extends $e{constructor(t=new j,e=new j){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=t,this.v2=e}getPoint(t,e=new j){let i=e;return t===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(t).add(this.v1)),i}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new j){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},nc=class extends $e{constructor(t=new L,e=new L){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=t,this.v2=e}getPoint(t,e=new L){let i=e;return t===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(t).add(this.v1)),i}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new L){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},qr=class extends $e{constructor(t=new j,e=new j,i=new j){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=t,this.v1=e,this.v2=i}getPoint(t,e=new j){let i=e,s=this.v0,r=this.v1,o=this.v2;return i.set(Ps(t,s.x,r.x,o.x),Ps(t,s.y,r.y,o.y)),i}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},ic=class extends $e{constructor(t=new L,e=new L,i=new L){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=t,this.v1=e,this.v2=i}getPoint(t,e=new L){let i=e,s=this.v0,r=this.v1,o=this.v2;return i.set(Ps(t,s.x,r.x,o.x),Ps(t,s.y,r.y,o.y),Ps(t,s.z,r.z,o.z)),i}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},Yr=class extends $e{constructor(t=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=t}getPoint(t,e=new j){let i=e,s=this.points,r=(s.length-1)*t,o=Math.floor(r),a=r-o,c=s[o===0?o:o-1],l=s[o],h=s[o>s.length-2?s.length-1:o+1],u=s[o>s.length-3?s.length-1:o+2];return i.set(ch(a,c.x,l.x,h.x,u.x),ch(a,c.y,l.y,h.y,u.y)),i}copy(t){super.copy(t),this.points=[];for(let e=0,i=t.points.length;e<i;e++){let s=t.points[e];this.points.push(s.clone())}return this}toJSON(){let t=super.toJSON();t.points=[];for(let e=0,i=this.points.length;e<i;e++){let s=this.points[e];t.points.push(s.toArray())}return t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,i=t.points.length;e<i;e++){let s=t.points[e];this.points.push(new j().fromArray(s))}return this}},sc=Object.freeze({__proto__:null,ArcCurve:ja,CatmullRomCurve3:tc,CubicBezierCurve:Wr,CubicBezierCurve3:ec,EllipseCurve:zs,LineCurve:Xr,LineCurve3:nc,QuadraticBezierCurve:qr,QuadraticBezierCurve3:ic,SplineCurve:Yr}),rc=class extends $e{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(t){this.curves.push(t)}closePath(){let t=this.curves[0].getPoint(0),e=this.curves[this.curves.length-1].getPoint(1);if(!t.equals(e)){let i=t.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new sc[i](e,t))}return this}getPoint(t,e){let i=t*this.getLength(),s=this.getCurveLengths(),r=0;for(;r<s.length;){if(s[r]>=i){let o=s[r]-i,a=this.curves[r],c=a.getLength(),l=c===0?0:1-o/c;return a.getPointAt(l,e)}r++}return null}getLength(){let t=this.getCurveLengths();return t[t.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;let t=[],e=0;for(let i=0,s=this.curves.length;i<s;i++)e+=this.curves[i].getLength(),t.push(e);return this.cacheLengths=t,t}getSpacedPoints(t=40){let e=[];for(let i=0;i<=t;i++)e.push(this.getPoint(i/t));return this.autoClose&&e.push(e[0]),e}getPoints(t=12){let e=[],i;for(let s=0,r=this.curves;s<r.length;s++){let o=r[s],a=o.isEllipseCurve?t*2:o.isLineCurve||o.isLineCurve3?1:o.isSplineCurve?t*o.points.length:t,c=o.getPoints(a);for(let l=0;l<c.length;l++){let h=c[l];i&&i.equals(h)||(e.push(h),i=h)}}return this.autoClose&&e.length>1&&!e[e.length-1].equals(e[0])&&e.push(e[0]),e}copy(t){super.copy(t),this.curves=[];for(let e=0,i=t.curves.length;e<i;e++){let s=t.curves[e];this.curves.push(s.clone())}return this.autoClose=t.autoClose,this}toJSON(){let t=super.toJSON();t.autoClose=this.autoClose,t.curves=[];for(let e=0,i=this.curves.length;e<i;e++){let s=this.curves[e];t.curves.push(s.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.autoClose=t.autoClose,this.curves=[];for(let e=0,i=t.curves.length;e<i;e++){let s=t.curves[e];this.curves.push(new sc[s.type]().fromJSON(s))}return this}},Ei=class extends rc{constructor(t){super(),this.type="Path",this.currentPoint=new j,t&&this.setFromPoints(t)}setFromPoints(t){this.moveTo(t[0].x,t[0].y);for(let e=1,i=t.length;e<i;e++)this.lineTo(t[e].x,t[e].y);return this}moveTo(t,e){return this.currentPoint.set(t,e),this}lineTo(t,e){let i=new Xr(this.currentPoint.clone(),new j(t,e));return this.curves.push(i),this.currentPoint.set(t,e),this}quadraticCurveTo(t,e,i,s){let r=new qr(this.currentPoint.clone(),new j(t,e),new j(i,s));return this.curves.push(r),this.currentPoint.set(i,s),this}bezierCurveTo(t,e,i,s,r,o){let a=new Wr(this.currentPoint.clone(),new j(t,e),new j(i,s),new j(r,o));return this.curves.push(a),this.currentPoint.set(r,o),this}splineThru(t){let e=[this.currentPoint.clone()].concat(t),i=new Yr(e);return this.curves.push(i),this.currentPoint.copy(t[t.length-1]),this}arc(t,e,i,s,r,o){let a=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(t+a,e+c,i,s,r,o),this}absarc(t,e,i,s,r,o){return this.absellipse(t,e,i,i,s,r,o),this}ellipse(t,e,i,s,r,o,a,c){let l=this.currentPoint.x,h=this.currentPoint.y;return this.absellipse(t+l,e+h,i,s,r,o,a,c),this}absellipse(t,e,i,s,r,o,a,c){let l=new zs(t,e,i,s,r,o,a,c);if(this.curves.length>0){let u=l.getPoint(0);u.equals(this.currentPoint)||this.lineTo(u.x,u.y)}this.curves.push(l);let h=l.getPoint(1);return this.currentPoint.copy(h),this}copy(t){return super.copy(t),this.currentPoint.copy(t.currentPoint),this}toJSON(){let t=super.toJSON();return t.currentPoint=this.currentPoint.toArray(),t}fromJSON(t){return super.fromJSON(t),this.currentPoint.fromArray(t.currentPoint),this}},oc=class n extends Te{constructor(t=[new j(0,-.5),new j(.5,0),new j(0,.5)],e=12,i=0,s=Math.PI*2){super(),this.type="LatheGeometry",this.parameters={points:t,segments:e,phiStart:i,phiLength:s},e=Math.floor(e),s=ye(s,0,Math.PI*2);let r=[],o=[],a=[],c=[],l=[],h=1/e,u=new L,d=new j,f=new L,g=new L,_=new L,m=0,p=0;for(let S=0;S<=t.length-1;S++)switch(S){case 0:m=t[S+1].x-t[S].x,p=t[S+1].y-t[S].y,f.x=p*1,f.y=-m,f.z=p*0,_.copy(f),f.normalize(),c.push(f.x,f.y,f.z);break;case t.length-1:c.push(_.x,_.y,_.z);break;default:m=t[S+1].x-t[S].x,p=t[S+1].y-t[S].y,f.x=p*1,f.y=-m,f.z=p*0,g.copy(f),f.x+=_.x,f.y+=_.y,f.z+=_.z,f.normalize(),c.push(f.x,f.y,f.z),_.copy(g)}for(let S=0;S<=e;S++){let M=i+S*h*s,x=Math.sin(M),D=Math.cos(M);for(let T=0;T<=t.length-1;T++){u.x=t[T].x*x,u.y=t[T].y,u.z=t[T].x*D,o.push(u.x,u.y,u.z),d.x=S/e,d.y=T/(t.length-1),a.push(d.x,d.y);let R=c[3*T+0]*x,C=c[3*T+1],b=c[3*T+0]*D;l.push(R,C,b)}}for(let S=0;S<e;S++)for(let M=0;M<t.length-1;M++){let x=M+S*t.length,D=x,T=x+t.length,R=x+t.length+1,C=x+1;r.push(D,T,C),r.push(R,C,T)}this.setIndex(r),this.setAttribute("position",new se(o,3)),this.setAttribute("uv",new se(a,2)),this.setAttribute("normal",new se(l,3))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.points,t.segments,t.phiStart,t.phiLength)}},ss=class n extends oc{constructor(t=1,e=1,i=4,s=8){let r=new Ei;r.absarc(0,-e/2,t,Math.PI*1.5,0),r.absarc(0,e/2,t,0,Math.PI*.5),super(r.getPoints(i),s),this.type="CapsuleGeometry",this.parameters={radius:t,length:e,capSegments:i,radialSegments:s}}static fromJSON(t){return new n(t.radius,t.length,t.capSegments,t.radialSegments)}};var fn=class n extends Te{constructor(t=1,e=1,i=1,s=32,r=1,o=!1,a=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:t,radiusBottom:e,height:i,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:c};let l=this;s=Math.floor(s),r=Math.floor(r);let h=[],u=[],d=[],f=[],g=0,_=[],m=i/2,p=0;S(),o===!1&&(t>0&&M(!0),e>0&&M(!1)),this.setIndex(h),this.setAttribute("position",new se(u,3)),this.setAttribute("normal",new se(d,3)),this.setAttribute("uv",new se(f,2));function S(){let x=new L,D=new L,T=0,R=(e-t)/i;for(let C=0;C<=r;C++){let b=[],y=C/r,I=y*(e-t)+t;for(let B=0;B<=s;B++){let N=B/s,H=N*c+a,Y=Math.sin(H),V=Math.cos(H);D.x=I*Y,D.y=-y*i+m,D.z=I*V,u.push(D.x,D.y,D.z),x.set(Y,R,V).normalize(),d.push(x.x,x.y,x.z),f.push(N,1-y),b.push(g++)}_.push(b)}for(let C=0;C<s;C++)for(let b=0;b<r;b++){let y=_[b][C],I=_[b+1][C],B=_[b+1][C+1],N=_[b][C+1];(t>0||b!==0)&&(h.push(y,I,N),T+=3),(e>0||b!==r-1)&&(h.push(I,B,N),T+=3)}l.addGroup(p,T,0),p+=T}function M(x){let D=g,T=new j,R=new L,C=0,b=x===!0?t:e,y=x===!0?1:-1;for(let B=1;B<=s;B++)u.push(0,m*y,0),d.push(0,y,0),f.push(.5,.5),g++;let I=g;for(let B=0;B<=s;B++){let H=B/s*c+a,Y=Math.cos(H),V=Math.sin(H);R.x=b*V,R.y=m*y,R.z=b*Y,u.push(R.x,R.y,R.z),d.push(0,y,0),T.x=Y*.5+.5,T.y=V*.5*y+.5,f.push(T.x,T.y),g++}for(let B=0;B<s;B++){let N=D+B,H=I+B;x===!0?h.push(H,H+1,N):h.push(H+1,H,N),C+=3}l.addGroup(p,C,x===!0?1:2),p+=C}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.radiusTop,t.radiusBottom,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}};var ac=class n extends Te{constructor(t=[],e=[],i=1,s=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:t,indices:e,radius:i,detail:s};let r=[],o=[];a(s),l(i),h(),this.setAttribute("position",new se(r,3)),this.setAttribute("normal",new se(r.slice(),3)),this.setAttribute("uv",new se(o,2)),s===0?this.computeVertexNormals():this.normalizeNormals();function a(S){let M=new L,x=new L,D=new L;for(let T=0;T<e.length;T+=3)f(e[T+0],M),f(e[T+1],x),f(e[T+2],D),c(M,x,D,S)}function c(S,M,x,D){let T=D+1,R=[];for(let C=0;C<=T;C++){R[C]=[];let b=S.clone().lerp(x,C/T),y=M.clone().lerp(x,C/T),I=T-C;for(let B=0;B<=I;B++)B===0&&C===T?R[C][B]=b:R[C][B]=b.clone().lerp(y,B/I)}for(let C=0;C<T;C++)for(let b=0;b<2*(T-C)-1;b++){let y=Math.floor(b/2);b%2===0?(d(R[C][y+1]),d(R[C+1][y]),d(R[C][y])):(d(R[C][y+1]),d(R[C+1][y+1]),d(R[C+1][y]))}}function l(S){let M=new L;for(let x=0;x<r.length;x+=3)M.x=r[x+0],M.y=r[x+1],M.z=r[x+2],M.normalize().multiplyScalar(S),r[x+0]=M.x,r[x+1]=M.y,r[x+2]=M.z}function h(){let S=new L;for(let M=0;M<r.length;M+=3){S.x=r[M+0],S.y=r[M+1],S.z=r[M+2];let x=m(S)/2/Math.PI+.5,D=p(S)/Math.PI+.5;o.push(x,1-D)}g(),u()}function u(){for(let S=0;S<o.length;S+=6){let M=o[S+0],x=o[S+2],D=o[S+4],T=Math.max(M,x,D),R=Math.min(M,x,D);T>.9&&R<.1&&(M<.2&&(o[S+0]+=1),x<.2&&(o[S+2]+=1),D<.2&&(o[S+4]+=1))}}function d(S){r.push(S.x,S.y,S.z)}function f(S,M){let x=S*3;M.x=t[x+0],M.y=t[x+1],M.z=t[x+2]}function g(){let S=new L,M=new L,x=new L,D=new L,T=new j,R=new j,C=new j;for(let b=0,y=0;b<r.length;b+=9,y+=6){S.set(r[b+0],r[b+1],r[b+2]),M.set(r[b+3],r[b+4],r[b+5]),x.set(r[b+6],r[b+7],r[b+8]),T.set(o[y+0],o[y+1]),R.set(o[y+2],o[y+3]),C.set(o[y+4],o[y+5]),D.copy(S).add(M).add(x).divideScalar(3);let I=m(D);_(T,y+0,S,I),_(R,y+2,M,I),_(C,y+4,x,I)}}function _(S,M,x,D){D<0&&S.x===1&&(o[M]=S.x-1),x.x===0&&x.z===0&&(o[M]=D/2/Math.PI+.5)}function m(S){return Math.atan2(S.z,-S.x)}function p(S){return Math.atan2(-S.y,Math.sqrt(S.x*S.x+S.z*S.z))}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.vertices,t.indices,t.radius,t.details)}};var Xe=class extends Ei{constructor(t){super(t),this.uuid=Ri(),this.type="Shape",this.holes=[]}getPointsHoles(t){let e=[];for(let i=0,s=this.holes.length;i<s;i++)e[i]=this.holes[i].getPoints(t);return e}extractPoints(t){return{shape:this.getPoints(t),holes:this.getPointsHoles(t)}}copy(t){super.copy(t),this.holes=[];for(let e=0,i=t.holes.length;e<i;e++){let s=t.holes[e];this.holes.push(s.clone())}return this}toJSON(){let t=super.toJSON();t.uuid=this.uuid,t.holes=[];for(let e=0,i=this.holes.length;e<i;e++){let s=this.holes[e];t.holes.push(s.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.uuid=t.uuid,this.holes=[];for(let e=0,i=t.holes.length;e<i;e++){let s=t.holes[e];this.holes.push(new Ei().fromJSON(s))}return this}},O0={triangulate:function(n,t,e=2){let i=t&&t.length,s=i?t[0]*e:n.length,r=kh(n,0,s,e,!0),o=[];if(!r||r.next===r.prev)return o;let a,c,l,h,u,d,f;if(i&&(r=V0(n,t,r,e)),n.length>80*e){a=l=n[0],c=h=n[1];for(let g=e;g<s;g+=e)u=n[g],d=n[g+1],u<a&&(a=u),d<c&&(c=d),u>l&&(l=u),d>h&&(h=d);f=Math.max(l-a,h-c),f=f!==0?32767/f:0}return Hs(r,o,e,a,c,f,0),o}};function kh(n,t,e,i,s){let r,o;if(s===j0(n,t,e,i)>0)for(r=t;r<e;r+=i)o=lh(r,n[r],n[r+1],o);else for(r=e-i;r>=t;r-=i)o=lh(r,n[r],n[r+1],o);return o&&ao(o,o.next)&&(Vs(o),o=o.next),o}function bi(n,t){if(!n)return n;t||(t=n);let e=n,i;do if(i=!1,!e.steiner&&(ao(e,e.next)||fe(e.prev,e,e.next)===0)){if(Vs(e),e=t=e.prev,e===e.next)break;i=!0}else e=e.next;while(i||e!==t);return t}function Hs(n,t,e,i,s,r,o){if(!n)return;!o&&r&&Y0(n,i,s,r);let a=n,c,l;for(;n.prev!==n.next;){if(c=n.prev,l=n.next,r?z0(n,i,s,r):B0(n)){t.push(c.i/e|0),t.push(n.i/e|0),t.push(l.i/e|0),Vs(n),n=l.next,a=l.next;continue}if(n=l,n===a){o?o===1?(n=H0(bi(n),t,e),Hs(n,t,e,i,s,r,2)):o===2&&k0(n,t,e,i,s,r):Hs(bi(n),t,e,i,s,r,1);break}}}function B0(n){let t=n.prev,e=n,i=n.next;if(fe(t,e,i)>=0)return!1;let s=t.x,r=e.x,o=i.x,a=t.y,c=e.y,l=i.y,h=s<r?s<o?s:o:r<o?r:o,u=a<c?a<l?a:l:c<l?c:l,d=s>r?s>o?s:o:r>o?r:o,f=a>c?a>l?a:l:c>l?c:l,g=i.next;for(;g!==t;){if(g.x>=h&&g.x<=d&&g.y>=u&&g.y<=f&&Yi(s,a,r,c,o,l,g.x,g.y)&&fe(g.prev,g,g.next)>=0)return!1;g=g.next}return!0}function z0(n,t,e,i){let s=n.prev,r=n,o=n.next;if(fe(s,r,o)>=0)return!1;let a=s.x,c=r.x,l=o.x,h=s.y,u=r.y,d=o.y,f=a<c?a<l?a:l:c<l?c:l,g=h<u?h<d?h:d:u<d?u:d,_=a>c?a>l?a:l:c>l?c:l,m=h>u?h>d?h:d:u>d?u:d,p=cc(f,g,t,e,i),S=cc(_,m,t,e,i),M=n.prevZ,x=n.nextZ;for(;M&&M.z>=p&&x&&x.z<=S;){if(M.x>=f&&M.x<=_&&M.y>=g&&M.y<=m&&M!==s&&M!==o&&Yi(a,h,c,u,l,d,M.x,M.y)&&fe(M.prev,M,M.next)>=0||(M=M.prevZ,x.x>=f&&x.x<=_&&x.y>=g&&x.y<=m&&x!==s&&x!==o&&Yi(a,h,c,u,l,d,x.x,x.y)&&fe(x.prev,x,x.next)>=0))return!1;x=x.nextZ}for(;M&&M.z>=p;){if(M.x>=f&&M.x<=_&&M.y>=g&&M.y<=m&&M!==s&&M!==o&&Yi(a,h,c,u,l,d,M.x,M.y)&&fe(M.prev,M,M.next)>=0)return!1;M=M.prevZ}for(;x&&x.z<=S;){if(x.x>=f&&x.x<=_&&x.y>=g&&x.y<=m&&x!==s&&x!==o&&Yi(a,h,c,u,l,d,x.x,x.y)&&fe(x.prev,x,x.next)>=0)return!1;x=x.nextZ}return!0}function H0(n,t,e){let i=n;do{let s=i.prev,r=i.next.next;!ao(s,r)&&Vh(s,i,i.next,r)&&ks(s,r)&&ks(r,s)&&(t.push(s.i/e|0),t.push(i.i/e|0),t.push(r.i/e|0),Vs(i),Vs(i.next),i=n=r),i=i.next}while(i!==n);return bi(i)}function k0(n,t,e,i,s,r){let o=n;do{let a=o.next.next;for(;a!==o.prev;){if(o.i!==a.i&&J0(o,a)){let c=Gh(o,a);o=bi(o,o.next),c=bi(c,c.next),Hs(o,t,e,i,s,r,0),Hs(c,t,e,i,s,r,0);return}a=a.next}o=o.next}while(o!==n)}function V0(n,t,e,i){let s=[],r,o,a,c,l;for(r=0,o=t.length;r<o;r++)a=t[r]*i,c=r<o-1?t[r+1]*i:n.length,l=kh(n,a,c,i,!1),l===l.next&&(l.steiner=!0),s.push($0(l));for(s.sort(G0),r=0;r<s.length;r++)e=W0(s[r],e);return e}function G0(n,t){return n.x-t.x}function W0(n,t){let e=X0(n,t);if(!e)return t;let i=Gh(e,n);return bi(i,i.next),bi(e,e.next)}function X0(n,t){let e=t,i=-1/0,s,r=n.x,o=n.y;do{if(o<=e.y&&o>=e.next.y&&e.next.y!==e.y){let d=e.x+(o-e.y)*(e.next.x-e.x)/(e.next.y-e.y);if(d<=r&&d>i&&(i=d,s=e.x<e.next.x?e:e.next,d===r))return s}e=e.next}while(e!==t);if(!s)return null;let a=s,c=s.x,l=s.y,h=1/0,u;e=s;do r>=e.x&&e.x>=c&&r!==e.x&&Yi(o<l?r:i,o,c,l,o<l?i:r,o,e.x,e.y)&&(u=Math.abs(o-e.y)/(r-e.x),ks(e,n)&&(u<h||u===h&&(e.x>s.x||e.x===s.x&&q0(s,e)))&&(s=e,h=u)),e=e.next;while(e!==a);return s}function q0(n,t){return fe(n.prev,n,t.prev)<0&&fe(t.next,n,n.next)<0}function Y0(n,t,e,i){let s=n;do s.z===0&&(s.z=cc(s.x,s.y,t,e,i)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==n);s.prevZ.nextZ=null,s.prevZ=null,Z0(s)}function Z0(n){let t,e,i,s,r,o,a,c,l=1;do{for(e=n,n=null,r=null,o=0;e;){for(o++,i=e,a=0,t=0;t<l&&(a++,i=i.nextZ,!!i);t++);for(c=l;a>0||c>0&&i;)a!==0&&(c===0||!i||e.z<=i.z)?(s=e,e=e.nextZ,a--):(s=i,i=i.nextZ,c--),r?r.nextZ=s:n=s,s.prevZ=r,r=s;e=i}r.nextZ=null,l*=2}while(o>1);return n}function cc(n,t,e,i,s){return n=(n-e)*s|0,t=(t-i)*s|0,n=(n|n<<8)&16711935,n=(n|n<<4)&252645135,n=(n|n<<2)&858993459,n=(n|n<<1)&1431655765,t=(t|t<<8)&16711935,t=(t|t<<4)&252645135,t=(t|t<<2)&858993459,t=(t|t<<1)&1431655765,n|t<<1}function $0(n){let t=n,e=n;do(t.x<e.x||t.x===e.x&&t.y<e.y)&&(e=t),t=t.next;while(t!==n);return e}function Yi(n,t,e,i,s,r,o,a){return(s-o)*(t-a)>=(n-o)*(r-a)&&(n-o)*(i-a)>=(e-o)*(t-a)&&(e-o)*(r-a)>=(s-o)*(i-a)}function J0(n,t){return n.next.i!==t.i&&n.prev.i!==t.i&&!K0(n,t)&&(ks(n,t)&&ks(t,n)&&Q0(n,t)&&(fe(n.prev,n,t.prev)||fe(n,t.prev,t))||ao(n,t)&&fe(n.prev,n,n.next)>0&&fe(t.prev,t,t.next)>0)}function fe(n,t,e){return(t.y-n.y)*(e.x-t.x)-(t.x-n.x)*(e.y-t.y)}function ao(n,t){return n.x===t.x&&n.y===t.y}function Vh(n,t,e,i){let s=vr(fe(n,t,e)),r=vr(fe(n,t,i)),o=vr(fe(e,i,n)),a=vr(fe(e,i,t));return!!(s!==r&&o!==a||s===0&&yr(n,e,t)||r===0&&yr(n,i,t)||o===0&&yr(e,n,i)||a===0&&yr(e,t,i))}function yr(n,t,e){return t.x<=Math.max(n.x,e.x)&&t.x>=Math.min(n.x,e.x)&&t.y<=Math.max(n.y,e.y)&&t.y>=Math.min(n.y,e.y)}function vr(n){return n>0?1:n<0?-1:0}function K0(n,t){let e=n;do{if(e.i!==n.i&&e.next.i!==n.i&&e.i!==t.i&&e.next.i!==t.i&&Vh(e,e.next,n,t))return!0;e=e.next}while(e!==n);return!1}function ks(n,t){return fe(n.prev,n,n.next)<0?fe(n,t,n.next)>=0&&fe(n,n.prev,t)>=0:fe(n,t,n.prev)<0||fe(n,n.next,t)<0}function Q0(n,t){let e=n,i=!1,s=(n.x+t.x)/2,r=(n.y+t.y)/2;do e.y>r!=e.next.y>r&&e.next.y!==e.y&&s<(e.next.x-e.x)*(r-e.y)/(e.next.y-e.y)+e.x&&(i=!i),e=e.next;while(e!==n);return i}function Gh(n,t){let e=new lc(n.i,n.x,n.y),i=new lc(t.i,t.x,t.y),s=n.next,r=t.prev;return n.next=t,t.prev=n,e.next=s,s.prev=e,i.next=e,e.prev=i,r.next=i,i.prev=r,i}function lh(n,t,e,i){let s=new lc(n,t,e);return i?(s.next=i.next,s.prev=i,i.next.prev=s,i.next=s):(s.prev=s,s.next=s),s}function Vs(n){n.next.prev=n.prev,n.prev.next=n.next,n.prevZ&&(n.prevZ.nextZ=n.nextZ),n.nextZ&&(n.nextZ.prevZ=n.prevZ)}function lc(n,t,e){this.i=n,this.x=t,this.y=e,this.prev=null,this.next=null,this.z=0,this.prevZ=null,this.nextZ=null,this.steiner=!1}function j0(n,t,e,i){let s=0;for(let r=t,o=e-i;r<e;r+=i)s+=(n[o]-n[r])*(n[r+1]+n[o+1]),o=r;return s}var Zn=class n{static area(t){let e=t.length,i=0;for(let s=e-1,r=0;r<e;s=r++)i+=t[s].x*t[r].y-t[r].x*t[s].y;return i*.5}static isClockWise(t){return n.area(t)<0}static triangulateShape(t,e){let i=[],s=[],r=[];hh(t),uh(i,t);let o=t.length;e.forEach(hh);for(let c=0;c<e.length;c++)s.push(o),o+=e[c].length,uh(i,e[c]);let a=O0.triangulate(i,s);for(let c=0;c<a.length;c+=3)r.push(a.slice(c,c+3));return r}};function hh(n){let t=n.length;t>2&&n[t-1].equals(n[0])&&n.pop()}function uh(n,t){for(let e=0;e<t.length;e++)n.push(t[e].x),n.push(t[e].y)}var jn=class n extends Te{constructor(t=new Xe([new j(.5,.5),new j(-.5,.5),new j(-.5,-.5),new j(.5,-.5)]),e={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:t,options:e},t=Array.isArray(t)?t:[t];let i=this,s=[],r=[];for(let a=0,c=t.length;a<c;a++){let l=t[a];o(l)}this.setAttribute("position",new se(s,3)),this.setAttribute("uv",new se(r,2)),this.computeVertexNormals();function o(a){let c=[],l=e.curveSegments!==void 0?e.curveSegments:12,h=e.steps!==void 0?e.steps:1,u=e.depth!==void 0?e.depth:1,d=e.bevelEnabled!==void 0?e.bevelEnabled:!0,f=e.bevelThickness!==void 0?e.bevelThickness:.2,g=e.bevelSize!==void 0?e.bevelSize:f-.1,_=e.bevelOffset!==void 0?e.bevelOffset:0,m=e.bevelSegments!==void 0?e.bevelSegments:3,p=e.extrudePath,S=e.UVGenerator!==void 0?e.UVGenerator:t_,M,x=!1,D,T,R,C;p&&(M=p.getSpacedPoints(h),x=!0,d=!1,D=p.computeFrenetFrames(h,!1),T=new L,R=new L,C=new L),d||(m=0,f=0,g=0,_=0);let b=a.extractPoints(l),y=b.shape,I=b.holes;if(!Zn.isClockWise(y)){y=y.reverse();for(let Z=0,st=I.length;Z<st;Z++){let A=I[Z];Zn.isClockWise(A)&&(I[Z]=A.reverse())}}let N=Zn.triangulateShape(y,I),H=y;for(let Z=0,st=I.length;Z<st;Z++){let A=I[Z];y=y.concat(A)}function Y(Z,st,A){return st||console.error("THREE.ExtrudeGeometry: vec does not exist"),Z.clone().addScaledVector(st,A)}let V=y.length,rt=N.length;function X(Z,st,A){let St,ot,Mt,ht=Z.x-st.x,Ut=Z.y-st.y,xt=A.x-Z.x,w=A.y-Z.y,v=ht*ht+Ut*Ut,z=ht*w-Ut*xt;if(Math.abs(z)>Number.EPSILON){let $=Math.sqrt(v),et=Math.sqrt(xt*xt+w*w),J=st.x-Ut/$,Tt=st.y+ht/$,ft=A.x-w/et,yt=A.y+xt/et,Zt=((ft-J)*w-(yt-Tt)*xt)/(ht*w-Ut*xt);St=J+ht*Zt-Z.x,ot=Tt+Ut*Zt-Z.y;let at=St*St+ot*ot;if(at<=2)return new j(St,ot);Mt=Math.sqrt(at/2)}else{let $=!1;ht>Number.EPSILON?xt>Number.EPSILON&&($=!0):ht<-Number.EPSILON?xt<-Number.EPSILON&&($=!0):Math.sign(Ut)===Math.sign(w)&&($=!0),$?(St=-Ut,ot=ht,Mt=Math.sqrt(v)):(St=ht,ot=Ut,Mt=Math.sqrt(v/2))}return new j(St/Mt,ot/Mt)}let it=[];for(let Z=0,st=H.length,A=st-1,St=Z+1;Z<st;Z++,A++,St++)A===st&&(A=0),St===st&&(St=0),it[Z]=X(H[Z],H[A],H[St]);let dt=[],vt,zt=it.concat();for(let Z=0,st=I.length;Z<st;Z++){let A=I[Z];vt=[];for(let St=0,ot=A.length,Mt=ot-1,ht=St+1;St<ot;St++,Mt++,ht++)Mt===ot&&(Mt=0),ht===ot&&(ht=0),vt[St]=X(A[St],A[Mt],A[ht]);dt.push(vt),zt=zt.concat(vt)}for(let Z=0;Z<m;Z++){let st=Z/m,A=f*Math.cos(st*Math.PI/2),St=g*Math.sin(st*Math.PI/2)+_;for(let ot=0,Mt=H.length;ot<Mt;ot++){let ht=Y(H[ot],it[ot],St);tt(ht.x,ht.y,-A)}for(let ot=0,Mt=I.length;ot<Mt;ot++){let ht=I[ot];vt=dt[ot];for(let Ut=0,xt=ht.length;Ut<xt;Ut++){let w=Y(ht[Ut],vt[Ut],St);tt(w.x,w.y,-A)}}}let Kt=g+_;for(let Z=0;Z<V;Z++){let st=d?Y(y[Z],zt[Z],Kt):y[Z];x?(R.copy(D.normals[0]).multiplyScalar(st.x),T.copy(D.binormals[0]).multiplyScalar(st.y),C.copy(M[0]).add(R).add(T),tt(C.x,C.y,C.z)):tt(st.x,st.y,0)}for(let Z=1;Z<=h;Z++)for(let st=0;st<V;st++){let A=d?Y(y[st],zt[st],Kt):y[st];x?(R.copy(D.normals[Z]).multiplyScalar(A.x),T.copy(D.binormals[Z]).multiplyScalar(A.y),C.copy(M[Z]).add(R).add(T),tt(C.x,C.y,C.z)):tt(A.x,A.y,u/h*Z)}for(let Z=m-1;Z>=0;Z--){let st=Z/m,A=f*Math.cos(st*Math.PI/2),St=g*Math.sin(st*Math.PI/2)+_;for(let ot=0,Mt=H.length;ot<Mt;ot++){let ht=Y(H[ot],it[ot],St);tt(ht.x,ht.y,u+A)}for(let ot=0,Mt=I.length;ot<Mt;ot++){let ht=I[ot];vt=dt[ot];for(let Ut=0,xt=ht.length;Ut<xt;Ut++){let w=Y(ht[Ut],vt[Ut],St);x?tt(w.x,w.y+M[h-1].y,M[h-1].x+A):tt(w.x,w.y,u+A)}}}P(),k();function P(){let Z=s.length/3;if(d){let st=0,A=V*st;for(let St=0;St<rt;St++){let ot=N[St];wt(ot[2]+A,ot[1]+A,ot[0]+A)}st=h+m*2,A=V*st;for(let St=0;St<rt;St++){let ot=N[St];wt(ot[0]+A,ot[1]+A,ot[2]+A)}}else{for(let st=0;st<rt;st++){let A=N[st];wt(A[2],A[1],A[0])}for(let st=0;st<rt;st++){let A=N[st];wt(A[0]+V*h,A[1]+V*h,A[2]+V*h)}}i.addGroup(Z,s.length/3-Z,0)}function k(){let Z=s.length/3,st=0;nt(H,st),st+=H.length;for(let A=0,St=I.length;A<St;A++){let ot=I[A];nt(ot,st),st+=ot.length}i.addGroup(Z,s.length/3-Z,1)}function nt(Z,st){let A=Z.length;for(;--A>=0;){let St=A,ot=A-1;ot<0&&(ot=Z.length-1);for(let Mt=0,ht=h+m*2;Mt<ht;Mt++){let Ut=V*Mt,xt=V*(Mt+1),w=st+St+Ut,v=st+ot+Ut,z=st+ot+xt,$=st+St+xt;Dt(w,v,z,$)}}}function tt(Z,st,A){c.push(Z),c.push(st),c.push(A)}function wt(Z,st,A){Lt(Z),Lt(st),Lt(A);let St=s.length/3,ot=S.generateTopUV(i,s,St-3,St-2,St-1);Yt(ot[0]),Yt(ot[1]),Yt(ot[2])}function Dt(Z,st,A,St){Lt(Z),Lt(st),Lt(St),Lt(st),Lt(A),Lt(St);let ot=s.length/3,Mt=S.generateSideWallUV(i,s,ot-6,ot-3,ot-2,ot-1);Yt(Mt[0]),Yt(Mt[1]),Yt(Mt[3]),Yt(Mt[1]),Yt(Mt[2]),Yt(Mt[3])}function Lt(Z){s.push(c[Z*3+0]),s.push(c[Z*3+1]),s.push(c[Z*3+2])}function Yt(Z){r.push(Z.x),r.push(Z.y)}}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}toJSON(){let t=super.toJSON(),e=this.parameters.shapes,i=this.parameters.options;return e_(e,i,t)}static fromJSON(t,e){let i=[];for(let r=0,o=t.shapes.length;r<o;r++){let a=e[t.shapes[r]];i.push(a)}let s=t.options.extrudePath;return s!==void 0&&(t.options.extrudePath=new sc[s.type]().fromJSON(s)),new n(i,t.options)}},t_={generateTopUV:function(n,t,e,i,s){let r=t[e*3],o=t[e*3+1],a=t[i*3],c=t[i*3+1],l=t[s*3],h=t[s*3+1];return[new j(r,o),new j(a,c),new j(l,h)]},generateSideWallUV:function(n,t,e,i,s,r){let o=t[e*3],a=t[e*3+1],c=t[e*3+2],l=t[i*3],h=t[i*3+1],u=t[i*3+2],d=t[s*3],f=t[s*3+1],g=t[s*3+2],_=t[r*3],m=t[r*3+1],p=t[r*3+2];return Math.abs(a-h)<Math.abs(o-l)?[new j(o,1-c),new j(l,1-u),new j(d,1-g),new j(_,1-p)]:[new j(a,1-c),new j(h,1-u),new j(f,1-g),new j(m,1-p)]}};function e_(n,t,e){if(e.shapes=[],Array.isArray(n))for(let i=0,s=n.length;i<s;i++){let r=n[i];e.shapes.push(r.uuid)}else e.shapes.push(n.uuid);return e.options=Object.assign({},t),t.extrudePath!==void 0&&(e.options.extrudePath=t.extrudePath.toJSON()),e}var rs=class n extends ac{constructor(t=1,e=0){let i=(1+Math.sqrt(5))/2,s=[-1,i,0,1,i,0,-1,-i,0,1,-i,0,0,-1,i,0,1,i,0,-1,-i,0,1,-i,i,0,-1,i,0,1,-i,0,-1,-i,0,1],r=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(s,r,t,e),this.type="IcosahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new n(t.radius,t.detail)}};var wi=class n extends Te{constructor(t=new Xe([new j(0,.5),new j(-.5,-.5),new j(.5,-.5)]),e=12){super(),this.type="ShapeGeometry",this.parameters={shapes:t,curveSegments:e};let i=[],s=[],r=[],o=[],a=0,c=0;if(Array.isArray(t)===!1)l(t);else for(let h=0;h<t.length;h++)l(t[h]),this.addGroup(a,c,h),a+=c,c=0;this.setIndex(i),this.setAttribute("position",new se(s,3)),this.setAttribute("normal",new se(r,3)),this.setAttribute("uv",new se(o,2));function l(h){let u=s.length/3,d=h.extractPoints(e),f=d.shape,g=d.holes;Zn.isClockWise(f)===!1&&(f=f.reverse());for(let m=0,p=g.length;m<p;m++){let S=g[m];Zn.isClockWise(S)===!0&&(g[m]=S.reverse())}let _=Zn.triangulateShape(f,g);for(let m=0,p=g.length;m<p;m++){let S=g[m];f=f.concat(S)}for(let m=0,p=f.length;m<p;m++){let S=f[m];s.push(S.x,S.y,0),r.push(0,0,1),o.push(S.x,S.y)}for(let m=0,p=_.length;m<p;m++){let S=_[m],M=S[0]+u,x=S[1]+u,D=S[2]+u;i.push(M,x,D),c+=3}}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}toJSON(){let t=super.toJSON(),e=this.parameters.shapes;return n_(e,t)}static fromJSON(t,e){let i=[];for(let s=0,r=t.shapes.length;s<r;s++){let o=e[t.shapes[s]];i.push(o)}return new n(i,t.curveSegments)}};function n_(n,t){if(t.shapes=[],Array.isArray(n))for(let e=0,i=n.length;e<i;e++){let s=n[e];t.shapes.push(s.uuid)}else t.shapes.push(n.uuid);return t}var ti=class n extends Te{constructor(t=1,e=32,i=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:i,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},e=Math.max(3,Math.floor(e)),i=Math.max(2,Math.floor(i));let c=Math.min(o+a,Math.PI),l=0,h=[],u=new L,d=new L,f=[],g=[],_=[],m=[];for(let p=0;p<=i;p++){let S=[],M=p/i,x=0;p===0&&o===0?x=.5/e:p===i&&c===Math.PI&&(x=-.5/e);for(let D=0;D<=e;D++){let T=D/e;u.x=-t*Math.cos(s+T*r)*Math.sin(o+M*a),u.y=t*Math.cos(o+M*a),u.z=t*Math.sin(s+T*r)*Math.sin(o+M*a),g.push(u.x,u.y,u.z),d.copy(u).normalize(),_.push(d.x,d.y,d.z),m.push(T+x,1-M),S.push(l++)}h.push(S)}for(let p=0;p<i;p++)for(let S=0;S<e;S++){let M=h[p][S+1],x=h[p][S],D=h[p+1][S],T=h[p+1][S+1];(p!==0||o>0)&&f.push(M,x,T),(p!==i-1||c<Math.PI)&&f.push(x,D,T)}this.setIndex(f),this.setAttribute("position",new se(g,3)),this.setAttribute("normal",new se(_,3)),this.setAttribute("uv",new se(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new n(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}};var pe=class extends Si{static get type(){return"MeshStandardMaterial"}constructor(t){super(),this.isMeshStandardMaterial=!0,this.defines={STANDARD:""},this.color=new kt(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new kt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Ph,this.normalScale=new j(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new un,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.envMapIntensity=t.envMapIntensity,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}},Zr=class extends pe{static get type(){return"MeshPhysicalMaterial"}constructor(t){super(),this.isMeshPhysicalMaterial=!0,this.defines={STANDARD:"",PHYSICAL:""},this.anisotropyRotation=0,this.anisotropyMap=null,this.clearcoatMap=null,this.clearcoatRoughness=0,this.clearcoatRoughnessMap=null,this.clearcoatNormalScale=new j(1,1),this.clearcoatNormalMap=null,this.ior=1.5,Object.defineProperty(this,"reflectivity",{get:function(){return ye(2.5*(this.ior-1)/(this.ior+1),0,1)},set:function(e){this.ior=(1+.4*e)/(1-.4*e)}}),this.iridescenceMap=null,this.iridescenceIOR=1.3,this.iridescenceThicknessRange=[100,400],this.iridescenceThicknessMap=null,this.sheenColor=new kt(0),this.sheenColorMap=null,this.sheenRoughness=1,this.sheenRoughnessMap=null,this.transmissionMap=null,this.thickness=0,this.thicknessMap=null,this.attenuationDistance=1/0,this.attenuationColor=new kt(1,1,1),this.specularIntensity=1,this.specularIntensityMap=null,this.specularColor=new kt(1,1,1),this.specularColorMap=null,this._anisotropy=0,this._clearcoat=0,this._dispersion=0,this._iridescence=0,this._sheen=0,this._transmission=0,this.setValues(t)}get anisotropy(){return this._anisotropy}set anisotropy(t){this._anisotropy>0!=t>0&&this.version++,this._anisotropy=t}get clearcoat(){return this._clearcoat}set clearcoat(t){this._clearcoat>0!=t>0&&this.version++,this._clearcoat=t}get iridescence(){return this._iridescence}set iridescence(t){this._iridescence>0!=t>0&&this.version++,this._iridescence=t}get dispersion(){return this._dispersion}set dispersion(t){this._dispersion>0!=t>0&&this.version++,this._dispersion=t}get sheen(){return this._sheen}set sheen(t){this._sheen>0!=t>0&&this.version++,this._sheen=t}get transmission(){return this._transmission}set transmission(t){this._transmission>0!=t>0&&this.version++,this._transmission=t}copy(t){return super.copy(t),this.defines={STANDARD:"",PHYSICAL:""},this.anisotropy=t.anisotropy,this.anisotropyRotation=t.anisotropyRotation,this.anisotropyMap=t.anisotropyMap,this.clearcoat=t.clearcoat,this.clearcoatMap=t.clearcoatMap,this.clearcoatRoughness=t.clearcoatRoughness,this.clearcoatRoughnessMap=t.clearcoatRoughnessMap,this.clearcoatNormalMap=t.clearcoatNormalMap,this.clearcoatNormalScale.copy(t.clearcoatNormalScale),this.dispersion=t.dispersion,this.ior=t.ior,this.iridescence=t.iridescence,this.iridescenceMap=t.iridescenceMap,this.iridescenceIOR=t.iridescenceIOR,this.iridescenceThicknessRange=[...t.iridescenceThicknessRange],this.iridescenceThicknessMap=t.iridescenceThicknessMap,this.sheen=t.sheen,this.sheenColor.copy(t.sheenColor),this.sheenColorMap=t.sheenColorMap,this.sheenRoughness=t.sheenRoughness,this.sheenRoughnessMap=t.sheenRoughnessMap,this.transmission=t.transmission,this.transmissionMap=t.transmissionMap,this.thickness=t.thickness,this.thicknessMap=t.thicknessMap,this.attenuationDistance=t.attenuationDistance,this.attenuationColor.copy(t.attenuationColor),this.specularIntensity=t.specularIntensity,this.specularIntensityMap=t.specularIntensityMap,this.specularColor.copy(t.specularColor),this.specularColorMap=t.specularColorMap,this}};function Mr(n,t,e){return!n||!e&&n.constructor===t?n:typeof t.BYTES_PER_ELEMENT=="number"?new t(n):Array.prototype.slice.call(n)}function i_(n){return ArrayBuffer.isView(n)&&!(n instanceof DataView)}var os=class{constructor(t,e,i,s){this.parameterPositions=t,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new e.constructor(i),this.sampleValues=e,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(t){let e=this.parameterPositions,i=this._cachedIndex,s=e[i],r=e[i-1];n:{t:{let o;e:{i:if(!(t<s)){for(let a=i+2;;){if(s===void 0){if(t<r)break i;return i=e.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===a)break;if(r=s,s=e[++i],t<s)break t}o=e.length;break e}if(!(t>=r)){let a=e[1];t<a&&(i=2,r=a);for(let c=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===c)break;if(s=r,r=e[--i-1],t>=r)break t}o=i,i=0;break e}break n}for(;i<o;){let a=i+o>>>1;t<e[a]?o=a:i=a+1}if(s=e[i],r=e[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=e.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,t,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(t){let e=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=t*s;for(let o=0;o!==s;++o)e[o]=i[r+o];return e}interpolate_(){throw new Error("call to abstract method")}intervalChanged_(){}},hc=class extends os{constructor(t,e,i,s){super(t,e,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:ul,endingEnd:ul}}intervalChanged_(t,e,i){let s=this.parameterPositions,r=t-2,o=t+1,a=s[r],c=s[o];if(a===void 0)switch(this.getSettings_().endingStart){case dl:r=t,a=2*e-i;break;case fl:r=s.length-2,a=e+s[r]-s[r+1];break;default:r=t,a=i}if(c===void 0)switch(this.getSettings_().endingEnd){case dl:o=t,c=2*i-e;break;case fl:o=1,c=i+s[1]-s[0];break;default:o=t-1,c=e}let l=(i-e)*.5,h=this.valueSize;this._weightPrev=l/(e-a),this._weightNext=l/(c-i),this._offsetPrev=r*h,this._offsetNext=o*h}interpolate_(t,e,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=t*a,l=c-a,h=this._offsetPrev,u=this._offsetNext,d=this._weightPrev,f=this._weightNext,g=(i-e)/(s-e),_=g*g,m=_*g,p=-d*m+2*d*_-d*g,S=(1+d)*m+(-1.5-2*d)*_+(-.5+d)*g+1,M=(-1-f)*m+(1.5+f)*_+.5*g,x=f*m-f*_;for(let D=0;D!==a;++D)r[D]=p*o[h+D]+S*o[l+D]+M*o[c+D]+x*o[u+D];return r}},uc=class extends os{constructor(t,e,i,s){super(t,e,i,s)}interpolate_(t,e,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=t*a,l=c-a,h=(i-e)/(s-e),u=1-h;for(let d=0;d!==a;++d)r[d]=o[l+d]*u+o[c+d]*h;return r}},dc=class extends os{constructor(t,e,i,s){super(t,e,i,s)}interpolate_(t){return this.copySampleValue_(t-1)}},on=class{constructor(t,e,i,s){if(t===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(e===void 0||e.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+t);this.name=t,this.times=Mr(e,this.TimeBufferType),this.values=Mr(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(t){let e=t.constructor,i;if(e.toJSON!==this.toJSON)i=e.toJSON(t);else{i={name:t.name,times:Mr(t.times,Array),values:Mr(t.values,Array)};let s=t.getInterpolation();s!==t.DefaultInterpolation&&(i.interpolation=s)}return i.type=t.ValueTypeName,i}InterpolantFactoryMethodDiscrete(t){return new dc(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodLinear(t){return new uc(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodSmooth(t){return new hc(this.times,this.values,this.getValueSize(),t)}setInterpolation(t){let e;switch(t){case Ar:e=this.InterpolantFactoryMethodDiscrete;break;case Fa:e=this.InterpolantFactoryMethodLinear;break;case vo:e=this.InterpolantFactoryMethodSmooth;break}if(e===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(t!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return console.warn("THREE.KeyframeTrack:",i),this}return this.createInterpolant=e,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return Ar;case this.InterpolantFactoryMethodLinear:return Fa;case this.InterpolantFactoryMethodSmooth:return vo}}getValueSize(){return this.values.length/this.times.length}shift(t){if(t!==0){let e=this.times;for(let i=0,s=e.length;i!==s;++i)e[i]+=t}return this}scale(t){if(t!==1){let e=this.times;for(let i=0,s=e.length;i!==s;++i)e[i]*=t}return this}trim(t,e){let i=this.times,s=i.length,r=0,o=s-1;for(;r!==s&&i[r]<t;)++r;for(;o!==-1&&i[o]>e;)--o;if(++o,r!==0||o!==s){r>=o&&(o=Math.max(o,1),r=o-1);let a=this.getValueSize();this.times=i.slice(r,o),this.values=this.values.slice(r*a,o*a)}return this}validate(){let t=!0,e=this.getValueSize();e-Math.floor(e)!==0&&(console.error("THREE.KeyframeTrack: Invalid value size in track.",this),t=!1);let i=this.times,s=this.values,r=i.length;r===0&&(console.error("THREE.KeyframeTrack: Track is empty.",this),t=!1);let o=null;for(let a=0;a!==r;a++){let c=i[a];if(typeof c=="number"&&isNaN(c)){console.error("THREE.KeyframeTrack: Time is not a valid number.",this,a,c),t=!1;break}if(o!==null&&o>c){console.error("THREE.KeyframeTrack: Out of order keys.",this,a,c,o),t=!1;break}o=c}if(s!==void 0&&i_(s))for(let a=0,c=s.length;a!==c;++a){let l=s[a];if(isNaN(l)){console.error("THREE.KeyframeTrack: Value is not a valid number.",this,a,l),t=!1;break}}return t}optimize(){let t=this.times.slice(),e=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===vo,r=t.length-1,o=1;for(let a=1;a<r;++a){let c=!1,l=t[a],h=t[a+1];if(l!==h&&(a!==1||l!==t[0]))if(s)c=!0;else{let u=a*i,d=u-i,f=u+i;for(let g=0;g!==i;++g){let _=e[u+g];if(_!==e[d+g]||_!==e[f+g]){c=!0;break}}}if(c){if(a!==o){t[o]=t[a];let u=a*i,d=o*i;for(let f=0;f!==i;++f)e[d+f]=e[u+f]}++o}}if(r>0){t[o]=t[r];for(let a=r*i,c=o*i,l=0;l!==i;++l)e[c+l]=e[a+l];++o}return o!==t.length?(this.times=t.slice(0,o),this.values=e.slice(0,o*i)):(this.times=t,this.values=e),this}clone(){let t=this.times.slice(),e=this.values.slice(),i=this.constructor,s=new i(this.name,t,e);return s.createInterpolant=this.createInterpolant,s}};on.prototype.TimeBufferType=Float32Array;on.prototype.ValueBufferType=Float32Array;on.prototype.DefaultInterpolation=Fa;var Ti=class extends on{constructor(t,e,i){super(t,e,i)}};Ti.prototype.ValueTypeName="bool";Ti.prototype.ValueBufferType=Array;Ti.prototype.DefaultInterpolation=Ar;Ti.prototype.InterpolantFactoryMethodLinear=void 0;Ti.prototype.InterpolantFactoryMethodSmooth=void 0;var fc=class extends on{};fc.prototype.ValueTypeName="color";var pc=class extends on{};pc.prototype.ValueTypeName="number";var mc=class extends os{constructor(t,e,i,s){super(t,e,i,s)}interpolate_(t,e,i,s){let r=this.resultBuffer,o=this.sampleValues,a=this.valueSize,c=(i-e)/(s-e),l=t*a;for(let h=l+a;l!==h;l+=4)Kn.slerpFlat(r,0,o,l-a,o,l,c);return r}},$r=class extends on{InterpolantFactoryMethodLinear(t){return new mc(this.times,this.values,this.getValueSize(),t)}};$r.prototype.ValueTypeName="quaternion";$r.prototype.InterpolantFactoryMethodSmooth=void 0;var Ai=class extends on{constructor(t,e,i){super(t,e,i)}};Ai.prototype.ValueTypeName="string";Ai.prototype.ValueBufferType=Array;Ai.prototype.DefaultInterpolation=Ar;Ai.prototype.InterpolantFactoryMethodLinear=void 0;Ai.prototype.InterpolantFactoryMethodSmooth=void 0;var gc=class extends on{};gc.prototype.ValueTypeName="vector";var _c=class{constructor(t,e,i){let s=this,r=!1,o=0,a=0,c,l=[];this.onStart=void 0,this.onLoad=t,this.onProgress=e,this.onError=i,this.itemStart=function(h){a++,r===!1&&s.onStart!==void 0&&s.onStart(h,o,a),r=!0},this.itemEnd=function(h){o++,s.onProgress!==void 0&&s.onProgress(h,o,a),o===a&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return c?c(h):h},this.setURLModifier=function(h){return c=h,this},this.addHandler=function(h,u){return l.push(h,u),this},this.removeHandler=function(h){let u=l.indexOf(h);return u!==-1&&l.splice(u,2),this},this.getHandler=function(h){for(let u=0,d=l.length;u<d;u+=2){let f=l[u],g=l[u+1];if(f.global&&(f.lastIndex=0),f.test(h))return g}return null}}},s_=new _c,xc=class{constructor(t){this.manager=t!==void 0?t:s_,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={}}load(){}loadAsync(t,e){let i=this;return new Promise(function(s,r){i.load(t,s,e,r)})}parse(){}setCrossOrigin(t){return this.crossOrigin=t,this}setWithCredentials(t){return this.withCredentials=t,this}setPath(t){return this.path=t,this}setResourcePath(t){return this.resourcePath=t,this}setRequestHeader(t){return this.requestHeader=t,this}};xc.DEFAULT_MATERIAL_NAME="__DEFAULT";var as=class extends Ue{constructor(t,e=1){super(),this.isLight=!0,this.type="Light",this.color=new kt(t),this.intensity=e}dispose(){}copy(t,e){return super.copy(t,e),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){let e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,this.groundColor!==void 0&&(e.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(e.object.distance=this.distance),this.angle!==void 0&&(e.object.angle=this.angle),this.decay!==void 0&&(e.object.decay=this.decay),this.penumbra!==void 0&&(e.object.penumbra=this.penumbra),this.shadow!==void 0&&(e.object.shadow=this.shadow.toJSON()),this.target!==void 0&&(e.object.target=this.target.uuid),e}},Jr=class extends as{constructor(t,e,i){super(t,i),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(Ue.DEFAULT_UP),this.updateMatrix(),this.groundColor=new kt(e)}copy(t,e){return super.copy(t,e),this.groundColor.copy(t.groundColor),this}},Jo=new de,dh=new L,fh=new L,Kr=class{constructor(t){this.camera=t,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new j(512,512),this.map=null,this.mapPass=null,this.matrix=new de,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Bs,this._frameExtents=new j(1,1),this._viewportCount=1,this._viewports=[new ie(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(t){let e=this.camera,i=this.matrix;dh.setFromMatrixPosition(t.matrixWorld),e.position.copy(dh),fh.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(fh),e.updateMatrixWorld(),Jo.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Jo),i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(Jo)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.intensity=t.intensity,this.bias=t.bias,this.radius=t.radius,this.mapSize.copy(t.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){let t={};return this.intensity!==1&&(t.intensity=this.intensity),this.bias!==0&&(t.bias=this.bias),this.normalBias!==0&&(t.normalBias=this.normalBias),this.radius!==1&&(t.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(t.mapSize=this.mapSize.toArray()),t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}};var ph=new de,ws=new L,Ko=new L,yc=class extends Kr{constructor(){super(new Pe(90,1,.5,500)),this.isPointLightShadow=!0,this._frameExtents=new j(4,2),this._viewportCount=6,this._viewports=[new ie(2,1,1,1),new ie(0,1,1,1),new ie(3,1,1,1),new ie(1,1,1,1),new ie(3,0,1,1),new ie(1,0,1,1)],this._cubeDirections=[new L(1,0,0),new L(-1,0,0),new L(0,0,1),new L(0,0,-1),new L(0,1,0),new L(0,-1,0)],this._cubeUps=[new L(0,1,0),new L(0,1,0),new L(0,1,0),new L(0,1,0),new L(0,0,1),new L(0,0,-1)]}updateMatrices(t,e=0){let i=this.camera,s=this.matrix,r=t.distance||i.far;r!==i.far&&(i.far=r,i.updateProjectionMatrix()),ws.setFromMatrixPosition(t.matrixWorld),i.position.copy(ws),Ko.copy(i.position),Ko.add(this._cubeDirections[e]),i.up.copy(this._cubeUps[e]),i.lookAt(Ko),i.updateMatrixWorld(),s.makeTranslation(-ws.x,-ws.y,-ws.z),ph.multiplyMatrices(i.projectionMatrix,i.matrixWorldInverse),this._frustum.setFromProjectionMatrix(ph)}},Qr=class extends as{constructor(t,e,i=0,s=2){super(t,e),this.isPointLight=!0,this.type="PointLight",this.distance=i,this.decay=s,this.shadow=new yc}get power(){return this.intensity*4*Math.PI}set power(t){this.intensity=t/(4*Math.PI)}dispose(){this.shadow.dispose()}copy(t,e){return super.copy(t,e),this.distance=t.distance,this.decay=t.decay,this.shadow=t.shadow.clone(),this}},vc=class extends Kr{constructor(){super(new Br(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},jr=class extends as{constructor(t,e){super(t,e),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(Ue.DEFAULT_UP),this.updateMatrix(),this.target=new Ue,this.shadow=new vc}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}},to=class extends as{constructor(t,e){super(t,e),this.isAmbientLight=!0,this.type="AmbientLight"}};var eo=class{constructor(t=!0){this.autoStart=t,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1}start(){this.startTime=mh(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let t=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){let e=mh();t=(e-this.oldTime)/1e3,this.oldTime=e,this.elapsedTime+=t}return t}};function mh(){return performance.now()}var Nc="\\[\\]\\.:\\/",r_=new RegExp("["+Nc+"]","g"),Fc="[^"+Nc+"]",o_="[^"+Nc.replace("\\.","")+"]",a_=/((?:WC+[\/:])*)/.source.replace("WC",Fc),c_=/(WCOD+)?/.source.replace("WCOD",o_),l_=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Fc),h_=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Fc),u_=new RegExp("^"+a_+c_+l_+h_+"$"),d_=["material","materials","bones","map"],Mc=class{constructor(t,e,i){let s=i||ue.parseTrackName(e);this._targetGroup=t,this._bindings=t.subscribe_(e,s)}getValue(t,e){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(t,e)}setValue(t,e){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(t,e)}bind(){let t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,i=t.length;e!==i;++e)t[e].bind()}unbind(){let t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,i=t.length;e!==i;++e)t[e].unbind()}},ue=class n{constructor(t,e,i){this.path=e,this.parsedPath=i||n.parseTrackName(e),this.node=n.findNode(t,this.parsedPath.nodeName),this.rootNode=t,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(t,e,i){return t&&t.isAnimationObjectGroup?new n.Composite(t,e,i):new n(t,e,i)}static sanitizeNodeName(t){return t.replace(/\s/g,"_").replace(r_,"")}static parseTrackName(t){let e=u_.exec(t);if(e===null)throw new Error("PropertyBinding: Cannot parse trackName: "+t);let i={nodeName:e[2],objectName:e[3],objectIndex:e[4],propertyName:e[5],propertyIndex:e[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);d_.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("PropertyBinding: can not parse propertyName from trackName: "+t);return i}static findNode(t,e){if(e===void 0||e===""||e==="."||e===-1||e===t.name||e===t.uuid)return t;if(t.skeleton){let i=t.skeleton.getBoneByName(e);if(i!==void 0)return i}if(t.children){let i=function(r){for(let o=0;o<r.length;o++){let a=r[o];if(a.name===e||a.uuid===e)return a;let c=i(a.children);if(c)return c}return null},s=i(t.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(t,e){t[e]=this.targetObject[this.propertyName]}_getValue_array(t,e){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)t[e++]=i[s]}_getValue_arrayElement(t,e){t[e]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(t,e){this.resolvedProperty.toArray(t,e)}_setValue_direct(t,e){this.targetObject[this.propertyName]=t[e]}_setValue_direct_setNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(t,e){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=t[e++]}_setValue_array_setNeedsUpdate(t,e){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=t[e++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(t,e){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=t[e++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(t,e){this.resolvedProperty[this.propertyIndex]=t[e]}_setValue_arrayElement_setNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(t,e){this.resolvedProperty.fromArray(t,e)}_setValue_fromArray_setNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(t,e){this.bind(),this.getValue(t,e)}_setValue_unbound(t,e){this.bind(),this.setValue(t,e)}bind(){let t=this.node,e=this.parsedPath,i=e.objectName,s=e.propertyName,r=e.propertyIndex;if(t||(t=n.findNode(this.rootNode,e.nodeName),this.node=t),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!t){console.warn("THREE.PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let l=e.objectIndex;switch(i){case"materials":if(!t.material){console.error("THREE.PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!t.material.materials){console.error("THREE.PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}t=t.material.materials;break;case"bones":if(!t.skeleton){console.error("THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}t=t.skeleton.bones;for(let h=0;h<t.length;h++)if(t[h].name===l){l=h;break}break;case"map":if("map"in t){t=t.map;break}if(!t.material){console.error("THREE.PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!t.material.map){console.error("THREE.PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}t=t.material.map;break;default:if(t[i]===void 0){console.error("THREE.PropertyBinding: Can not bind to objectName of node undefined.",this);return}t=t[i]}if(l!==void 0){if(t[l]===void 0){console.error("THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,t);return}t=t[l]}}let o=t[s];if(o===void 0){let l=e.nodeName;console.error("THREE.PropertyBinding: Trying to update property for track: "+l+"."+s+" but it wasn't found.",t);return}let a=this.Versioning.None;this.targetObject=t,t.needsUpdate!==void 0?a=this.Versioning.NeedsUpdate:t.matrixWorldNeedsUpdate!==void 0&&(a=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!t.geometry){console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!t.geometry.morphAttributes){console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}t.morphTargetDictionary[r]!==void 0&&(r=t.morphTargetDictionary[r])}c=this.BindingType.ArrayElement,this.resolvedProperty=o,this.propertyIndex=r}else o.fromArray!==void 0&&o.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=o):Array.isArray(o)?(c=this.BindingType.EntireArray,this.resolvedProperty=o):this.propertyName=s;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][a]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};ue.Composite=Mc;ue.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};ue.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};ue.prototype.GetterByBindingType=[ue.prototype._getValue_direct,ue.prototype._getValue_array,ue.prototype._getValue_arrayElement,ue.prototype._getValue_toArray];ue.prototype.SetterByBindingTypeAndVersioning=[[ue.prototype._setValue_direct,ue.prototype._setValue_direct_setNeedsUpdate,ue.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[ue.prototype._setValue_array,ue.prototype._setValue_array_setNeedsUpdate,ue.prototype._setValue_array_setMatrixWorldNeedsUpdate],[ue.prototype._setValue_arrayElement,ue.prototype._setValue_arrayElement_setNeedsUpdate,ue.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[ue.prototype._setValue_fromArray,ue.prototype._setValue_fromArray_setNeedsUpdate,ue.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var C_=new Float32Array(1);var gh=new de,no=class{constructor(t,e,i=0,s=1/0){this.ray=new Lr(t,e),this.near=i,this.far=s,this.camera=null,this.layers=new Os,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(t,e){this.ray.set(t,e)}setFromCamera(t,e){e.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(t.x,t.y,.5).unproject(e).sub(this.ray.origin).normalize(),this.camera=e):e.isOrthographicCamera?(this.ray.origin.set(t.x,t.y,(e.near+e.far)/(e.near-e.far)).unproject(e),this.ray.direction.set(0,0,-1).transformDirection(e.matrixWorld),this.camera=e):console.error("THREE.Raycaster: Unsupported camera type: "+e.type)}setFromXRController(t){return gh.identity().extractRotation(t.matrixWorld),this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(gh),this}intersectObject(t,e=!0,i=[]){return Sc(t,this,i,e),i.sort(_h),i}intersectObjects(t,e=!0,i=[]){for(let s=0,r=t.length;s<r;s++)Sc(t[s],this,i,e);return i.sort(_h),i}};function _h(n,t){return n.distance-t.distance}function Sc(n,t,e,i){let s=!0;if(n.layers.test(t.layers)&&n.raycast(t,e)===!1&&(s=!1),s===!0&&i===!0){let r=n.children;for(let o=0,a=r.length;o<a;o++)Sc(r[o],t,e,!0)}}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Ec}}));typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Ec);function Xh(n,t=!1){let e=n[0].index!==null,i=new Set(Object.keys(n[0].attributes)),s=new Set(Object.keys(n[0].morphAttributes)),r={},o={},a=n[0].morphTargetsRelative,c=new Te,l=0;for(let h=0;h<n.length;++h){let u=n[h],d=0;if(e!==(u.index!==null))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+". All geometries must have compatible attributes; make sure index attribute exists among all geometries, or in none of them."),null;for(let f in u.attributes){if(!i.has(f))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+'. All geometries must have compatible attributes; make sure "'+f+'" attribute exists among all geometries, or in none of them.'),null;r[f]===void 0&&(r[f]=[]),r[f].push(u.attributes[f]),d++}if(d!==i.size)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+". Make sure all geometries have the same number of attributes."),null;if(a!==u.morphTargetsRelative)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+". .morphTargetsRelative must be consistent throughout all geometries."),null;for(let f in u.morphAttributes){if(!s.has(f))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+".  .morphAttributes must be consistent throughout all geometries."),null;o[f]===void 0&&(o[f]=[]),o[f].push(u.morphAttributes[f])}if(t){let f;if(e)f=u.index.count;else if(u.attributes.position!==void 0)f=u.attributes.position.count;else return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+h+". The geometry must have either an index or a position attribute"),null;c.addGroup(l,f,h),l+=f}}if(e){let h=0,u=[];for(let d=0;d<n.length;++d){let f=n[d].index;for(let g=0;g<f.count;++g)u.push(f.getX(g)+h);h+=n[d].attributes.position.count}c.setIndex(u)}for(let h in r){let u=Wh(r[h]);if(!u)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed while trying to merge the "+h+" attribute."),null;c.setAttribute(h,u)}for(let h in o){let u=o[h][0].length;if(u===0)break;c.morphAttributes=c.morphAttributes||{},c.morphAttributes[h]=[];for(let d=0;d<u;++d){let f=[];for(let _=0;_<o[h].length;++_)f.push(o[h][_][d]);let g=Wh(f);if(!g)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed while trying to merge the "+h+" morphAttribute."),null;c.morphAttributes[h].push(g)}}return c}function Wh(n){let t,e,i,s=-1,r=0;for(let l=0;l<n.length;++l){let h=n[l];if(t===void 0&&(t=h.array.constructor),t!==h.array.constructor)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.array must be of consistent array types across matching attributes."),null;if(e===void 0&&(e=h.itemSize),e!==h.itemSize)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.itemSize must be consistent across matching attributes."),null;if(i===void 0&&(i=h.normalized),i!==h.normalized)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.normalized must be consistent across matching attributes."),null;if(s===-1&&(s=h.gpuType),s!==h.gpuType)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.gpuType must be consistent across matching attributes."),null;r+=h.count*e}let o=new t(r),a=new we(o,e,i),c=0;for(let l=0;l<n.length;++l){let h=n[l];if(h.isInterleavedBufferAttribute){let u=c/e;for(let d=0,f=h.count;d<f;d++)for(let g=0;g<e;g++){let _=h.getComponent(d,g);a.setComponent(d+u,g,_)}}else o.set(h.array,c);c+=h.count*e}return s!==void 0&&(a.gpuType=s),a}function Ws(n,t,e,i){let s=n.attributes.uv;for(let r=0;r<s.count;r++){let o=Math.floor(r/4),a,c;o<2?(a=i,c=e):o<4?(a=t,c=i):(a=t,c=e),s.setXY(r,s.getX(r)*a,s.getY(r)*c)}return s.needsUpdate=!0,n}var co=class{constructor(t){this.scene=t,this.colliders=[],this.surfaces=[],this.occluders=[],this.groundY=-2.5}add(t,{occlude:e=!0,cast:i=!0,receive:s=!0}={}){return t.traverse(r=>{r.isMesh&&(r.castShadow=i,r.receiveShadow=s)}),this.scene.add(t),e&&this.occluders.push(t),t}collider(t,e,i,s,r,o){this.colliders.push({minX:Math.min(t,e),maxX:Math.max(t,e),minY:Math.min(i,s),maxY:Math.max(i,s),minZ:Math.min(r,o),maxZ:Math.max(r,o)})}colliderFromObject(t,e=0){t.updateMatrixWorld(!0);let i=new Dn().setFromObject(t);this.collider(i.min.x-e,i.max.x+e,i.min.y,i.max.y,i.min.z-e,i.max.z+e)}box(t,e,i,s,r,o,a,c={}){let l=Math.abs(e-t),h=Math.abs(s-i),u=Math.abs(o-r);if(l<.001||h<.001||u<.001)return null;let d=Ws(new ge(l,h,u),l,h,u),f=new Pt(d,a);return f.position.set((t+e)/2,(i+s)/2,(r+o)/2),this.add(f,c),c.collide&&this.collider(t,e,i,s,r,o),f}slopeZ(t,e,i,s,r,o,a,c,l={}){let h=i-t,u=s-e,d=Math.hypot(h,u),f=o-r,g=Ws(new ge(f,a,d),f,a,d),_=new Pt(g,c),m=Math.atan2(-u,h);_.rotation.x=m;let p=Math.cos(m)<0?-1:1,S=p*Math.cos(m),M=p*Math.sin(m);return _.position.set((r+o)/2,(e+s)/2+S*a/2,(t+i)/2+M*a/2),this.add(_,l)}slopeX(t,e,i,s,r,o,a,c,l={}){let h=i-t,u=s-e,d=Math.hypot(h,u),f=o-r,g=Ws(new ge(d,a,f),d,a,f),_=new Pt(g,c),m=Math.atan2(u,h);_.rotation.z=m;let p=Math.cos(m)<0?-1:1;return _.position.set((t+i)/2-p*Math.sin(m)*a/2,(e+s)/2+p*Math.cos(m)*a/2,(r+o)/2),this.add(_,l)}surface(t,e,i,s,r){this.surfaces.push({minX:Math.min(t,e),maxX:Math.max(t,e),minZ:Math.min(i,s),maxZ:Math.max(i,s),h:r})}floorAt(t,e,i,s=1.3){let r=this.groundY;for(let o of this.surfaces){if(t<o.minX||t>o.maxX||e<o.minZ||e>o.maxZ)continue;let a=typeof o.h=="function"?o.h(t,e):o.h;a<=i+s&&a>r&&(r=a)}return r}wall({axis:t,at:e,from:i,to:s,y0:r=0,y1:o=8,t:a=.4,matPos:c,matNeg:l,edge:h,openings:u=[],trim:d,glass:f,collide:g=!0,trimBoth:_=!0}){let m=Math.min(i,s),p=Math.max(i,s);h=h||l;let S=t==="x"?[h,h,h,h,c,l]:[c,l,h,h,h,h],M=(T,R,C,b)=>{R-T<.001||b-C<.001||(t==="x"?this.box(T,R,C,b,e-a/2,e+a/2,S,{collide:g}):this.box(e-a/2,e+a/2,C,b,T,R,S,{collide:g}))},x=[...u].sort((T,R)=>T.a-R.a),D=m;for(let T of x){M(D,T.a,r,o);let R=T.bottom??r,C=T.top??r+6.8;M(T.a,T.b,r,R),M(T.a,T.b,C,o),d&&this.openingTrim(t,e,T.a,T.b,R,C,a,d,!!T.window,_),T.window&&f&&this.windowGlass(t,e,T.a,T.b,R,C,f,d),D=T.b}M(D,p,r,o)}openingTrim(t,e,i,s,r,o,a,c,l,h=!0){let d=a+.12,f=h?e-d/2:e,g=e+d/2,_=(m,p,S,M)=>{t==="x"?this.box(m,p,S,M,f,g,c,{occlude:!1}):this.box(f,g,S,M,m,p,c,{occlude:!1})};_(i-.28,i,r-(l?.28:0),o+.28),_(s,s+.28,r-(l?.28:0),o+.28),_(i-.28,s+.28,o,o+.28),l&&(t==="x"?()=>this.box(i-.42000000000000004,s+.42000000000000004,r-.12,r,e-d/2-.15,e+d/2+.15,c,{occlude:!1}):()=>this.box(e-d/2-.15,e+d/2+.15,r-.12,r,i-.42000000000000004,s+.42000000000000004,c,{occlude:!1}))()}windowGlass(t,e,i,s,r,o,a,c){let h=(g,_,m,p,S,M)=>{let x={occlude:S===a,cast:S!==a};t==="x"?this.box(g,_,m,p,e-M,e+M,S,x):this.box(e-M,e+M,m,p,g,_,S,x)};if(h(i,s,r,o,a,.04),!c)return;let u=(r+o)/2;h(i,s,u-.08,u+.08,c,.09);let d=s-i,f=d>4.5?Math.round(d/2.6):d>3.2?2:1;for(let g=1;g<f;g++){let _=i+d*g/f;h(_-.1,_+.1,r,o,c,.09)}}optimize(){let t=new Set;for(let s of this.occluders)s.traverse(r=>r.isMesh&&t.add(r));let e=new Map,i=[];this.scene.updateMatrixWorld(!0),this.scene.traverse(s=>{if(!s.isMesh)return;i.push(s);let r=s.geometry.index?s.geometry.toNonIndexed():s.geometry.clone();for(let c of Object.keys(r.attributes))["position","normal","uv"].includes(c)||r.deleteAttribute(c);r.attributes.uv||r.setAttribute("uv",new we(new Float32Array(r.attributes.position.count*2),2)),r.applyMatrix4(s.matrixWorld);let o=Array.isArray(s.material)?s.material:[s.material],a=Array.isArray(s.material)&&r.groups.length?r.groups:[{start:0,count:r.attributes.position.count,materialIndex:0}];for(let c of a){let l=o[c.materialIndex??0];if(!l)continue;let h=`${l.uuid}|${s.castShadow}|${s.receiveShadow}|${t.has(s)}`;e.has(h)||e.set(h,{material:l,cast:s.castShadow,receive:s.receiveShadow,occlude:t.has(s),parts:[]});let u=new Te;for(let d of["position","normal","uv"]){let f=r.attributes[d];u.setAttribute(d,new we(f.array.slice(c.start*f.itemSize,(c.start+c.count)*f.itemSize),f.itemSize))}e.get(h).parts.push(u)}r.dispose()});for(let s of i)s.removeFromParent();this.occluders=[];for(let{material:s,cast:r,receive:o,occlude:a,parts:c}of e.values()){let l=Xh(c,!1);if(c.forEach(u=>u.dispose()),!l)continue;l.computeBoundingSphere(),l.computeBoundingBox();let h=new Pt(l,s);h.castShadow=r,h.receiveShadow=o,h.matrixAutoUpdate=!1,this.scene.add(h),a&&this.occluders.push(h)}}railing(t,e,i,s,r,o,a,{spacing:c=.45,collide:l=!0}={}){let h=Math.hypot(i-t,s-e),u=new Ot,d=Math.atan2(i-t,s-e),f=new Pt(new ge(.22,.14,h),a);f.position.set(0,o,h/2),u.add(f);let g=new Pt(new ge(.18,.1,h),a);g.position.set(0,.25,h/2),u.add(g);let _=Math.max(1,Math.floor(h/c)),m=new ge(.1,o-.25,.1);for(let S=1;S<_;S++){let M=new Pt(m,a);M.position.set(0,(o+.25)/2,h*S/_),u.add(M)}let p=new ge(.35,o+.25,.35);for(let S of[0,h]){let M=new Pt(p,a);M.position.set(0,(o+.25)/2,S),u.add(M)}return u.position.set(t,r,e),u.rotation.y=d,this.add(u,{occlude:!1}),l&&this.collider(Math.min(t,i)-.15,Math.max(t,i)+.15,r,r+o+.3,Math.min(e,s)-.15,Math.max(e,s)+.15),u}};function qh(n){let t=n>>>0;return()=>(t=t*1664525+1013904223>>>0,t/4294967296)}function ze(n,t,e,i=1){let s=document.createElement("canvas");s.width=s.height=n;let r=s.getContext("2d");e(r,n,qh(i));let o=new Qn(s);return o.wrapS=o.wrapT=Ds,o.colorSpace=xe,o.anisotropy=8,o.repeat.set(1/t,1/t),o}function ei(n,t,e,i,s=.06,r=!0){for(let o=0;o<i;o++){let a=e()<.5?0:255;n.fillStyle=`rgba(${a},${a},${a},${s*e()})`;let c=1+e()*2;n.fillRect(e()*t,e()*t,c,c)}}function Xs(n,t){let e=new kt(n);return e.offsetHSL(0,0,t),`#${e.getHexString()}`}function Yh(n="#e6eaeb"){return ze(512,4,(t,e,i)=>{t.fillStyle=n,t.fillRect(0,0,e,e);let s=8,r=e/s;for(let o=0;o<s;o++){let a=o*r,c=-i()*40;for(;c<e;){let h=30+i()*60;t.fillStyle=Xs(n,(i()-.5)*.05),t.fillRect(c,a,h,r),t.fillStyle="rgba(0,0,0,0.18)",t.fillRect(c,a,2,r);for(let u=0;u<4;u++)t.fillStyle="rgba(0,0,0,0.035)",t.fillRect(c+i()*h,a,1,r);c+=h}let l=t.createLinearGradient(0,a+r-7,0,a+r);l.addColorStop(0,"rgba(0,0,0,0)"),l.addColorStop(1,"rgba(0,0,0,0.28)"),t.fillStyle=l,t.fillRect(0,a+r-7,e,7)}ei(t,e,i,3e3,.08)},7)}function Zh(n="#f1f3f2"){return ze(256,2,(t,e,i)=>{t.fillStyle=n,t.fillRect(0,0,e,e);let s=6,r=e/s;for(let o=0;o<s;o++){let a=t.createLinearGradient(0,o*r,0,(o+1)*r);a.addColorStop(0,"rgba(0,0,0,0.0)"),a.addColorStop(.85,"rgba(0,0,0,0.05)"),a.addColorStop(1,"rgba(0,0,0,0.25)"),t.fillStyle=a,t.fillRect(0,o*r,e,r)}ei(t,e,i,600,.05)},3)}function $h(n="#6f7378"){return ze(512,4,(t,e,i)=>{t.fillStyle=Xs(n,-.08),t.fillRect(0,0,e,e);let s=10,r=e/s;for(let o=0;o<s;o++){let a=-i()*50;for(;a<e;){let c=25+i()*55;t.fillStyle=Xs(n,(i()-.5)*.12),t.fillRect(a+1,o*r,c-2,r-3),a+=c}t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,o*r+r-3,e,3)}for(let o=0;o<9e3;o++){let a=Math.floor(i()*255);t.fillStyle=`rgba(${a},${a},${a},0.12)`,t.fillRect(i()*e,i()*e,1.5,1.5)}},11)}function Jh(){return ze(512,4,(n,t,e)=>{n.fillStyle="#b9b2a6",n.fillRect(0,0,t,t);let i=18,s=t/6,r=t/i;for(let o=0;o<i;o++){let a=o%2?s/2:0;for(let c=-s;c<t+s;c+=s){let l=2+e()*8,h=30+e()*10;n.fillStyle=`hsl(${l}, ${45+e()*15}%, ${h}%)`,n.fillRect(c+a+2,o*r+2,s-4,r-4)}}ei(n,t,e,6e3,.12)},3)}function Kh(n="#d3a067"){return ze(512,4,(t,e,i)=>{let r=e/21;for(let o=0;o<21;o++){let a=-i()*200;for(;a<e;){let c=80+i()*260;t.fillStyle=Xs(n,(i()-.5)*.12),t.fillRect(o*r,a,r,c);for(let l=0;l<6;l++)t.fillStyle=`rgba(90,50,20,${.05+i()*.08})`,t.fillRect(o*r+i()*r,a,1,c);t.fillStyle="rgba(60,30,10,0.35)",t.fillRect(o*r,a,r,1.5),a+=c}t.fillStyle="rgba(60,30,10,0.3)",t.fillRect(o*r,0,1,e)}},5)}function Oc(n){return ze(256,3,(t,e,i)=>{t.fillStyle=n,t.fillRect(0,0,e,e),ei(t,e,i,14e3,.14)},13)}function Bc(n="#ebe5d8",t="#cfc7b6",e=4){return ze(256,4,(i,s,r)=>{i.fillStyle=t,i.fillRect(0,0,s,s);let o=s/e;for(let a=0;a<e;a++)for(let c=0;c<e;c++)i.fillStyle=Xs(n,(r()-.5)*.04),i.fillRect(a*o+1.5,c*o+1.5,o-3,o-3);ei(i,s,r,2e3,.05)},17)}function Qh(){return ze(512,8,(n,t,e)=>{n.fillStyle="#4f8a31",n.fillRect(0,0,t,t);for(let i=0;i<26e3;i++){let s=90+e()*80;n.fillStyle=`rgba(${40+e()*40},${s},${20+e()*30},0.5)`,n.fillRect(e()*t,e()*t,1.5,3+e()*3)}},19)}function jh(n="#c9c5bc"){return ze(256,6,(t,e,i)=>{t.fillStyle=n,t.fillRect(0,0,e,e),ei(t,e,i,9e3,.1),t.fillStyle="rgba(0,0,0,0.15)",t.fillRect(0,0,e,2)},23)}function tu(){return ze(256,8,(n,t,e)=>{n.fillStyle="#4a4b4d",n.fillRect(0,0,t,t),ei(n,t,e,16e3,.2)},29)}function eu(){return ze(256,4,(n,t,e)=>{n.fillStyle="#c8c3b8",n.fillRect(0,0,t,t);for(let i=0;i<5e3;i++){let s=150+e()*90;n.fillStyle=`rgb(${s},${s-5},${s-15})`,n.beginPath(),n.arc(e()*t,e()*t,1+e()*2.2,0,Math.PI*2),n.fill()}},31)}function nu(){return ze(512,8,(n,t,e)=>{n.fillStyle="#8c877c",n.fillRect(0,0,t,t);for(let i=0;i<90;i++){let s=e()*t,r=e()*t,o=25+e()*35,a=6+Math.floor(e()*3);n.fillStyle=`hsl(${30+e()*25}, ${8+e()*12}%, ${55+e()*15}%)`,n.beginPath();for(let c=0;c<a;c++){let l=c/a*Math.PI*2,h=o*(.7+e()*.3),u=s+Math.cos(l)*h,d=r+Math.sin(l)*h;c===0?n.moveTo(u,d):n.lineTo(u,d)}n.closePath(),n.fill()}ei(n,t,e,6e3,.1)},37)}function lo(n="#1f3557",t="#f4f4f0",e=8,i=!0){return ze(256,2,(s,r)=>{let o=r/e;for(let a=0;a<e;a++)s.fillStyle=a%2?t:n,i?s.fillRect(a*o,0,o,r):s.fillRect(0,a*o,r,o)})}function us(n="#e9dfcf"){return ze(256,3,(t,e,i)=>{let r=e/8,o=["#c9b79a","#8ea4bf","#d9c8b4","#b48d7a","#e8e0d0","#9bb39a"];for(let a=0;a<8;a++)for(let c=0;c<8;c++)t.fillStyle=o[Math.floor(i()*o.length)],t.fillRect(a*r,c*r,r,r),t.strokeStyle=n,t.lineWidth=2,t.strokeRect(a*r+3,c*r+3,r-6,r-6)},41)}function ni(n=3){let t=document.createElement("canvas");t.width=256,t.height=180;let e=t.getContext("2d"),i=qh(n),s=e.createLinearGradient(0,0,0,110);s.addColorStop(0,"#9cc3e4"),s.addColorStop(1,"#e8eef2"),e.fillStyle=s,e.fillRect(0,0,256,110),e.fillStyle="#4f7fa6",e.fillRect(0,100,256,30),e.fillStyle="#d8c39a",e.fillRect(0,125,256,55);for(let o=0;o<40;o++)e.fillStyle=`rgba(110,130,70,${.4+i()*.4})`,e.fillRect(i()*256,120+i()*50,2,-10-i()*15);e.fillStyle="#f2f2ee",e.fillRect(170,50,14,60),e.fillStyle="#b8413a",e.fillRect(168,42,18,10);let r=new Qn(t);return r.colorSpace=xe,r}function zc(n,t){let e=document.createElement("canvas");e.width=512,e.height=128;let i=e.getContext("2d");i.fillStyle="#1f3557",i.fillRect(0,0,512,128),i.fillStyle="#ffffff",i.font="bold 64px Georgia, serif",i.textAlign="center",i.textBaseline="middle",i.fillText(n,256,t?52:64),t&&(i.font="28px Georgia, serif",i.fillText(t,256,100));let s=new Qn(e);return s.colorSpace=xe,s}var Hc=new Map;function K(n,t={}){let e=`${n}|${JSON.stringify(t)}`;return Hc.has(e)||Hc.set(e,new pe({color:n,roughness:.8,...t})),Hc.get(e)}function lt(n,t,e,i,s,r,o,a){let c=Ws(new ge(t,e,i),t,e,i),l=new Pt(c,a);return l.position.set(s,r+e/2,o),n.add(l),l}function Ne(n,t,e,i,s,r,o,a,c=16){let l=new Pt(new fn(t,e,i,c),a);return l.position.set(s,r+i/2,o),n.add(l),l}function ce(n,t,e,i,s,r,{collide:o=!0,pad:a=0}={}){return t.position.set(e,i,s),t.rotation.y=r,n.add(t,{occlude:!1}),o&&n.colliderFromObject(t,a),t}var p_=()=>K("#8a5a35");function kc(n,t,e,i,{w:s=7,color:r="#24406b",y:o=0,pillows:a="#e9eef3",legs:c="#3b2a1f",armColor:l}={}){let h=new Ot,u=K(r),d=l?K(l):u;lt(h,s,.35,3,0,.3,0,u),lt(h,s-1,.55,2.4,0,.65,.25,u),lt(h,s,1.6,.7,0,.65,-1.15,u),lt(h,.6,1.2,3,-s/2+.3,.65,0,d),lt(h,.6,1.2,3,s/2-.3,.65,0,d);for(let f of[-1,1])for(let g of[-1,1])lt(h,.2,.3,.2,f*(s/2-.3),0,g*1.2,K(c));return lt(h,1.3,1.1,.35,-s/2+1.3,1.2,-.65,K(a)),lt(h,1.3,1.1,.35,s/2-1.3,1.2,-.65,K(a)),ce(n,h,t,o,e,i)}function qs(n,t,e,i,{color:s="#e8e2d4",stripe:r,y:o=0}={}){let a=new Ot,c=r?new pe({map:r,roughness:.9}):K(s);lt(a,2.6,.5,2.6,0,.9,0,c),lt(a,2.6,2.2,.45,0,1.2,-1.1,c),lt(a,.35,.9,2.4,-1.15,1.3,0,c),lt(a,.35,.9,2.4,1.15,1.3,0,c);for(let l of[-1,1])for(let h of[-1,1])lt(a,.18,.9,.18,l*1.1,0,h*1.1,p_());return ce(n,a,t,o,e,i)}function iu(n,t,e,i,{color:s="#8f9499"}={}){let r=new Ot,o=K(s);return lt(r,3,1.4,3,0,0,.2,o),lt(r,3,2.1,.9,0,1.4,-1.05,o),lt(r,.6,.8,3,-1.2,1.4,.2,o),lt(r,.6,.8,3,1.2,1.4,.2,o),ce(n,r,t,0,e,i)}function Vc(n,t,e,i,{w:s=6.5,y:r=0,cushion:o="#1f3a6b"}={}){let a=new Ot,c=K("#e3d6bb");lt(a,s,1,2.8,0,.3,0,c),lt(a,s,1.8,.4,0,.3,-1.2,c),lt(a,.4,1.9,2.8,-s/2+.2,.3,0,c),lt(a,.4,1.9,2.8,s/2-.2,.3,0,c);for(let l of[-1,1])for(let h of[-1,1])lt(a,.2,.3,.2,l*(s/2-.3),0,h*1.2,c);return lt(a,s-.9,.4,2.2,0,1.3,.15,K(o)),lt(a,s-.9,1.2,.35,0,1.6,-.85,K(o)),ce(n,a,t,r,e,i)}function ho(n,t,e,i,{y:s=0,stripe:r}={}){let o=new Ot,a=K("#e3d6bb");lt(o,2.4,1,2.4,0,.3,0,a),lt(o,2.4,1.7,.35,0,.3,-1.05,a),lt(o,.3,1.6,2.4,-1.05,.3,0,a),lt(o,.3,1.6,2.4,1.05,.3,0,a);let c=r?new pe({map:r}):K("#1f3a6b");return lt(o,1.8,.35,1.9,0,1.3,.1,c),lt(o,1.8,1.1,.3,0,1.6,-.75,c),ce(n,o,t,s,e,i)}function uo(n,t,e,i,{w:s=3,d:r=2,h:o=2.5,y:a=0,top:c="#8a5a35",legs:l,glass:h=!1}={}){let u=new Ot,d=h?K("#cfe3ea",{transparent:!0,opacity:.5}):K(c);lt(u,s,.12,r,0,o-.12,0,d);let f=K(l||c);for(let g of[-1,1])for(let _ of[-1,1])lt(u,.15,o-.12,.15,g*(s/2-.15),0,_*(r/2-.15),f);return ce(n,u,t,a,e,i)}function Gc(n,t,e,{r:i=1,h:s=2.2,top:r="#f2f0ea",y:o=0,collide:a=!0}={}){let c=new Ot;return Ne(c,i,i,.1,0,s-.1,0,K(r)),Ne(c,.1,.1,s-.1,0,0,0,K(r)),Ne(c,i*.5,i*.6,.1,0,0,0,K(r)),ce(n,c,t,o,e,0,{collide:a})}function Ys(n,t,e,i,{base:s="#6d9bb8",shade:r="#f7f2e6",light:o=!0}={}){let a=new Ot;return Ne(a,.25,.4,1.1,0,0,0,K(s,{roughness:.4})),Ne(a,.55,.75,.9,0,1.2,0,K(r,{emissive:o?"#ffdca8":"#000000",emissiveIntensity:.6,side:ve})),ce(n,a,t,e,i,0,{collide:!1})}function su(n,t,e,{y:i=0}={}){let s=new Ot;return Ne(s,.5,.6,.15,0,0,0,K("#c79a5d")),Ne(s,.15,.15,4.4,0,.15,0,K("#c79a5d")),Ne(s,.7,.85,1,0,4.5,0,K("#f6efdc",{emissive:"#ffdca8",emissiveIntensity:.7,side:ve})),ce(n,s,t,i,e,0,{collide:!1})}function m_(n,t,e,i,{color:s="#ece4d0",cushion:r="#2c3e5c",y:o=0}={}){let a=new Ot,c=K(s);lt(a,1.5,.12,1.5,0,1.45,0,c),lt(a,1.4,.12,1.4,0,1.57,0,K(r));for(let l of[-1,1])for(let h of[-1,1])lt(a,.1,1.45,.1,l*.62,0,h*.62,c);for(let l=-2;l<=2;l++)lt(a,.07,1.6,.07,l*.28,1.57,-.68,c);return lt(a,1.4,.16,.12,0,3.15,-.68,c),ce(n,a,t,o,e,i,{collide:!1})}function ru(n,t,e,i){let s=new Ot,r=K("#e9e0cc"),o=new Pt(new fn(1,1,.14,40),r);o.scale.set(3,1,1.9),o.position.y=2.45,s.add(o);for(let u of[-1,1])Ne(s,.14,.18,2.4,u*1.6,0,0,r);lt(s,3.6,.15,.3,0,.5,0,r),ce(n,s,t,0,e,i);let a=Math.cos(i),c=Math.sin(i),l=(u,d)=>[t+u*a+d*c,e-u*c+d*a],h=[[-1.4,1.9,0],[1.4,1.9,0],[-1.4,-1.9,Math.PI],[1.4,-1.9,Math.PI],[3.6,0,-Math.PI/2],[-3.6,0,Math.PI/2]];for(let[u,d,f]of h){let[g,_]=l(u,d);m_(n,g,_,i+f+Math.PI)}}function fo(n,t,e,{y:i=0}={}){let s=new Ot,r=K("#2d3540",{roughness:.5,metalness:.4});Ne(s,.6,.6,.15,0,2.6,0,r);for(let o=0;o<4;o++){let a=o/4*Math.PI*2+Math.PI/4;lt(s,.08,2.6,.08,Math.cos(a)*.45,0,Math.sin(a)*.45,r).rotation.set(Math.sin(a)*.1,0,-Math.cos(a)*.1)}return lt(s,.9,1,.1,0,2.75,-.5,r),ce(n,s,t,i,e,0,{collide:!1})}function ds(n,t,e,i,{w:s=5,l:r=6.8,quilt:o,sheet:a="#f3f1ea",frame:c="#6b4a30",y:l=0,pillows:h=2}={}){let u=new Ot;lt(u,s+.3,.9,r+.2,0,.3,0,K(c)),lt(u,s,.9,r,0,1.2,0,K(a));let d=o?new pe({map:o,roughness:.95}):K("#8fa9c4");lt(u,s+.15,.35,r*.7,0,1.95,r*.15+.05,d),lt(u,s+.3,3.4,.3,0,0,-r/2-.15,K(c));for(let f=0;f<h;f++){let g=h===1?0:(f-(h-1)/2)*(s/h);lt(u,s/h-.3,.45,1.2,g,2.1,-r/2+.9,K("#ffffff"))}return ce(n,u,t,l,e,i)}function an(n,t,e,i,{w:s=4,h:r=3,d:o=1.6,color:a="#7a5236",y:c=0}={}){let l=new Ot,h=K(a);lt(l,s,r,o,0,.2,0,h);let u=Math.max(2,Math.round(r/.9));for(let d=0;d<u;d++){let f=.35+d*(r-.3)/u;lt(l,s-.3,(r-.5)/u-.12,.05,0,f,o/2,K(new kt(a).offsetHSL(0,0,.04).getStyle())),lt(l,.4,.08,.08,0,f+.3,o/2+.05,K("#c9b17a",{metalness:.6,roughness:.4}))}return lt(l,s,.2,o,0,0,0,h),ce(n,l,t,c,e,i)}function Wc(n,t,e,i,s,{w:r=4,h:o=2.3}={}){let a=new Ot;return lt(a,r,o,.15,0,0,0,K("#111214",{roughness:.3})),lt(a,r-.15,o-.15,.02,0,.075,.08,K("#1d2633",{roughness:.1,emissive:"#0e1622",emissiveIntensity:.6})),ce(n,a,t,e,i,s,{collide:!1})}function ou(n,t,e,i,{color:s="#8f9aa1"}={}){let r=new Ot,o=K(s);return lt(r,3,3,1.6,0,0,0,o),lt(r,2.8,3.2,1.2,0,3,-.15,o),lt(r,2.4,2.7,.05,0,3.2,.47,K("#cfe0e8",{transparent:!0,opacity:.4})),lt(r,3.2,.2,1.4,0,6.2,-.1,o),ce(n,r,t,0,e,i)}function Xc(n,t,e,i,{y:s=0}={}){let r=new Ot,o=K("#fbfbf8",{roughness:.25});return Ne(r,.6,.45,1.3,0,0,.2,o),lt(r,1.6,1.3,.7,0,1.3,-.55,o),ce(n,r,t,s,e,i)}function qc(n,t,e,i,s,{y:r=0}={}){let o=new Ot,a=K("#fbfbf8",{roughness:.25}),c=e-t,l=s-i;return lt(o,c,1.8,l,0,0,0,a),lt(o,c-.5,.05,l-.5,0,1.76,0,K("#dfe9ee",{roughness:.1})),ce(n,o,(t+e)/2,r,(i+s)/2,0)}function Yc(n,t,e,i,{w:s=3,y:r=0,color:o="#f2f0ea"}={}){let a=new Ot;return lt(a,s,2.6,1.8,0,0,0,K(o)),lt(a,s+.1,.15,1.9,0,2.6,0,K("#f6f6f2",{roughness:.3})),lt(a,1.2,.05,.9,0,2.72,.1,K("#dfe5e8",{roughness:.1})),lt(a,s*.8,2.2,.08,0,3.6,-.86,K("#c9d8e0",{roughness:.05,metalness:.9})),ce(n,a,t,r,e,i)}function au(n,t,e,i){let s=new Ot,r=K("#f7f7f5",{roughness:.35});return lt(s,2.6,5.8,2.4,0,0,0,r),lt(s,2.62,.04,2.42,0,4.1,0,K("#c7c7c7")),lt(s,.1,1.2,.12,-1.1,4.4,1.25,K("#d9d9d9")),lt(s,.1,1.8,.12,-1.1,2,1.25,K("#d9d9d9")),ce(n,s,t,0,e,i)}function cu(n,t,e,i){let s=new Ot,r=K("#f5f5f3",{roughness:.35});lt(s,2.5,3,2.1,0,0,0,r),lt(s,2.5,.05,2.1,0,3,0,K("#2a2a2a",{roughness:.2}));for(let o of[-.6,.6])for(let a of[-.45,.45])Ne(s,.3,.3,.03,o,3.05,a,K("#111111"));return lt(s,2.5,.8,.2,0,3,-.95,r),lt(s,2.1,1.2,.05,0,.9,1.06,K("#1a1a1a",{roughness:.1})),ce(n,s,t,0,e,i)}function po(n,t,e,i,s,r,o){return n.box(t,e,r,r+.03,i,s,o,{occlude:!1,cast:!1})}function ii(n,t,e,i,s,r,o,a,c){let l=new Ot;lt(l,r+.25,o+.25,.1,0,-o/2-.125,0,K("#4a3020"));let h=new Pt(new Un(r,o),new pe({map:c,roughness:.6}));h.position.set(0,0,.06),l.add(h);let u,d,f;return t==="x"?(u=i,d=e+a*.08,f=a>0?0:Math.PI):(u=e+a*.08,d=i,f=a>0?Math.PI/2:-Math.PI/2),ce(n,l,u,s,d,f,{collide:!1})}function Fe(n,t,e,i,{r:s=.6,chandelier:r=!1}={}){let o=new Ot,a=K("#fff6e0",{emissive:"#ffe7b8",emissiveIntensity:1.2});return r?(Ne(o,.04,.04,1,0,-1,0,K("#b8934a",{metalness:.8,roughness:.3})),Ne(o,.9,.5,1,0,-2,0,K("#f3e3b5",{emissive:"#ffd890",emissiveIntensity:.8,transparent:!0,opacity:.9}),8)):Ne(o,s,s*.8,.35,0,-.35,0,a),ce(n,o,t,e,i,0,{collide:!1})}function Nn(n,t,e,{trunk:i=6,crown:s=7,color:r="#4f7f35",y:o=-2.5,lean:a=0,maple:c=!1,seed:l=1}={}){let h=new Ot,u=K("#5b4636",{roughness:1}),d=new Pt(new fn(i*.06,i*.09,i+s*.4,8),u);d.position.y=(i+s*.4)/2,d.rotation.z=a,h.add(d);let f=l,g=()=>(f=(f*9301+49297)%233280)/233280,_=c?["#7c8a3a","#9a6a2e","#a3472f","#5f7d33"]:[r,new kt(r).offsetHSL(0,0,.05).getStyle(),new kt(r).offsetHSL(.02,0,-.05).getStyle()],m=c?9:6;for(let p=0;p<m;p++){let S=s*(.35+g()*.25),M=new Pt(new rs(S,1),K(_[p%_.length],{flatShading:!0,roughness:1})),x=g()*Math.PI*2,D=s*.45*g();M.position.set(Math.cos(x)*D+a*-i,i+s*.3+g()*s*.5,Math.sin(x)*D),M.scale.y=c?.65:.9,h.add(M)}return h.position.set(t,o,e),n.add(h,{occlude:!1}),n.collider(t-i*.1,t+i*.1,o,o+i,e-i*.1,e+i*.1),h}function fs(n,t,e,{r:i=1.6,color:s="#3f6e2c",flowers:r,y:o=-2.5}={}){let a=new Ot,c=new Pt(new rs(i,1),K(s,{flatShading:!0,roughness:1}));if(c.scale.y=.75,c.position.y=i*.6,a.add(c),r)for(let l=0;l<6;l++){let h=new Pt(new rs(i*.22,0),K(r,{flatShading:!0})),u=l/6*Math.PI*2;h.position.set(Math.cos(u)*i*.75,i*.85,Math.sin(u)*i*.75),a.add(h)}return a.position.set(t,o,e),n.add(a,{occlude:!1}),a}function mo(n,t,e,i,s,r,o=-2.5){let a=K("#355f27",{roughness:1});n.box(t,e,o,o+r,i,s,a,{collide:!0,occlude:!1})}function lu(n,t,e,i,{y:s=-2.5}={}){let r=new Ot,o=K("#5d6166");lt(r,6,.15,2.4,0,2.4,0,o);for(let a of[-1.9,1.9])lt(r,6,.12,.9,0,1.45,a,o);for(let a of[-2.4,2.4])lt(r,.25,2.4,.25,a,0,-.8,o),lt(r,.25,2.4,.25,a,0,.8,o),lt(r,.25,.2,4.6,a,1.25,0,o);return ce(n,r,t,s,e,i)}function hu(n,t,e,{y:i=-2.5}={}){let s=new Ot,r=K("#e7ebec");lt(s,8,7,10,0,0,0,r),lt(s,3,6,.1,0,0,5.02,K("#f7f7f5"));let o=K("#62666b"),a=lt(s,5,.3,11,-2.1,7.6,0,o);a.rotation.z=.55;let c=lt(s,5,.3,11,2.1,7.6,0,o);c.rotation.z=-.55;let l=new Xe([new j(-4,0),new j(4,0),new j(0,2.4)]);for(let h of[-5,5]){let u=new Pt(new wi(l),new pe({color:"#e7ebec",side:ve}));u.position.set(0,7,h),s.add(u)}return ce(n,s,t,i,e,0)}function pn(n,t,e,{w:i=26,d:s=34,h:r=18,color:o="#dfe6ea",roof:a="#5d6168",rot:c=0}={}){let l=new Ot;lt(l,i,r,s,0,0,0,K(o));let h=new Xe([new j(-i/2-1,0),new j(i/2+1,0),new j(0,i*.38)]),u=new jn(h,{depth:s+2,bevelEnabled:!1}),d=new Pt(u,K(a,{flatShading:!0}));d.position.set(0,r,-s/2-1),l.add(d);let f=K("#3b4a57",{roughness:.2});for(let g of[3,11])for(let _ of[-i/3,0,i/3])lt(l,2.6,4,.1,_,g,s/2+.03,f),lt(l,.1,4,2.6,i/2+.03,g,_*.9,f),lt(l,.1,4,2.6,-i/2-.03,g,_*.9,f);return ce(n,l,t,-2.5,e,c)}var Zs=8.5/13,ps=.85,He=11.5,be=He-12*ps;function __(n){let t=Math.floor((He-n)/ps);return t<0?0:t>11?8.5:(t+1)*Zs}function Je(n){return K(n,{roughness:.92})}function mn(n,t="#ffffff",e={}){return new pe({map:n,color:t,roughness:.9,...e})}function uu(n){let t=Yh(),e={siding:mn(t),roof:mn($h()),brick:mn(Jh()),oak:mn(Kh(),"#ffffff",{roughness:.55}),trim:K("#f7f6f1",{roughness:.6}),ceiling:K("#f7f6f2",{roughness:.95}),concrete:mn(jh()),glass:new Zr({color:"#bcd6e6",transparent:!0,opacity:.22,roughness:.05,metalness:0,depthWrite:!1}),screen:new pe({color:"#2f3438",transparent:!0,opacity:.18,depthWrite:!1,side:ve}),shutter:K("#22324f",{roughness:.6}),kitchenTile:mn(Bc("#ece6da","#d5cdbd",4)),bathTile:mn(Bc("#f2f2ef","#c9cbc8",8)),beigeCarpet:mn(Oc("#cfc6b6")),blueCarpet:mn(Oc("#34507a")),porchFloor:K("#4d6a86",{roughness:.7}),maple:K("#dcc7a6",{roughness:.6}),counter:K("#f4f2ec",{roughness:.35}),clapboard:mn(Zh())},i={cream:Je("#efe7d4"),dining:Je("#a7c4e0"),kitchen:Je("#94b7d9"),br1:Je("#dcd6c8"),br2:Je("#d9d2c4"),br3:Je("#d8dcdc"),bath:Je("#cdd3d2"),upW:Je("#a9c3dd"),upE:Je("#e9e1d0"),bath2:Je("#d5dad8"),landing:Je("#efe7d4"),porch:Je("#f5f5f2")},s={y0:0,y1:8.5,t:.6,trim:e.trim,glass:e.glass,edge:e.trim},r={y0:0,y1:8,t:.4,trim:e.trim,glass:e.glass},o=(P,k,nt=2.6,tt=7)=>({a:P,b:k,bottom:nt,top:tt,window:!0}),a=(P,k,nt=6.9)=>({a:P,b:k,bottom:0,top:nt}),c=P=>({...P,bottom:(P.bottom??0)+8.5,top:(P.top??6.9)+8.5});n.box(-19.3,6,-2.5,0,-15.3,15.3,e.concrete,{collide:!0}),n.box(6,19.3,-2.5,0,-3,15.3,e.concrete,{collide:!0}),n.box(6,19.3,-2.5,-1,-15.3,-3,e.concrete,{collide:!0}),n.wall({...s,axis:"x",at:15,from:-19.3,to:-6,matPos:e.siding,matNeg:i.br1,openings:[o(-15.4,-12.6),o(-9.9,-7.1)]}),n.wall({...s,axis:"x",at:15,from:-6,to:6,matPos:e.siding,matNeg:i.cream,openings:[a(-1.5,1.5)]}),n.wall({...s,axis:"x",at:15,from:6,to:19.3,matPos:e.siding,matNeg:i.cream,openings:[{a:9,b:16,bottom:1.6,top:7}]}),n.wall({...s,axis:"x",at:-15,from:-19.3,to:-6,matPos:i.br2,matNeg:e.siding,openings:[o(-14.5,-11.5)]}),n.wall({...s,axis:"x",at:-15,from:-6,to:6,matPos:i.kitchen,matNeg:e.siding,openings:[o(-1.8,1.8,3.6,7)]}),n.wall({...s,axis:"z",at:-19,from:-14.7,to:-6,matPos:i.br2,matNeg:e.siding,openings:[o(-11.8,-9.2)]}),n.wall({...s,axis:"z",at:-19,from:-6,to:1,matPos:i.br3,matNeg:e.siding,openings:[o(-3.8,-1.2)]}),n.wall({...s,axis:"z",at:-19,from:1,to:6,matPos:i.bath,matNeg:e.siding,openings:[o(2.8,4.4,4.4,6.6)]}),n.wall({...s,axis:"z",at:-19,from:6,to:14.7,matPos:i.br1,matNeg:e.siding,openings:[o(9.2,11.8)]}),n.wall({...s,axis:"z",at:19,from:-3,to:14.7,matPos:e.siding,matNeg:i.cream,openings:[o(.4,3),o(8,10.6)]});let l=(P,k,nt,tt)=>n.box(P,k,nt,tt,15.3,15.42,e.shutter,{occlude:!1});for(let[P,k]of[[-15.4,-12.6],[-9.9,-7.1]])l(P-1.55,P-.35,2.4,7.2),l(k+.35,k+1.55,2.4,7.2),n.box(P-.1,k+.1,1.75,2.35,15.35,16.2,K("#3d4248"),{occlude:!1}),n.box(P+.05,k-.05,2.35,2.85,15.45,16.1,K("#4f8a36",{flatShading:!0}),{occlude:!1});l(7.35,8.55,1.6,7.2),l(16.45,17.65,1.6,7.2),n.box(9,16,.9,1.6,15.3,16.8,e.trim,{collide:!0}),n.box(8.8,16.2,7,7.5,15.3,17,e.trim),n.box(8.7,16.3,7.5,7.75,15.3,17.1,e.roof),n.windowGlass("x",16.65,9.2,15.8,1.6,7,e.glass,e.trim),n.windowGlass("z",9.1,15.3,16.7,1.6,7,e.glass,e.trim),n.windowGlass("z",15.9,15.3,16.7,1.6,7,e.glass,e.trim);for(let P of[9,16])n.box(P-.15,P+.15,1.6,7,16.55,16.85,e.trim,{occlude:!1});n.collider(9,16,0,8,15.3,16.9),n.wall({...r,axis:"z",at:-6,from:6,to:14.7,matPos:i.cream,matNeg:i.br1,openings:[a(7.5,10.2)]}),n.wall({...r,axis:"x",at:6,from:-18.7,to:-9,matPos:i.br1,matNeg:i.bath}),n.wall({...r,axis:"x",at:6,from:-9,to:-6,matPos:i.br1,matNeg:i.cream}),n.wall({...r,axis:"z",at:-9,from:1,to:6,matPos:i.cream,matNeg:i.bath,openings:[a(2,4.6)]}),n.wall({...r,axis:"x",at:1,from:-18.7,to:-9,matPos:i.bath,matNeg:i.br3}),n.wall({...r,axis:"z",at:-9,from:-6,to:1,matPos:i.cream,matNeg:i.br3,openings:[a(-2.6,0)]}),n.wall({...r,axis:"x",at:-6,from:-18.7,to:-9,matPos:i.br3,matNeg:i.br2}),n.wall({...r,axis:"x",at:-6,from:-9,to:-6,matPos:i.cream,matNeg:i.br2,openings:[a(-8.6,-6.4)]}),n.wall({...r,axis:"z",at:-6,from:-6,to:6,matPos:i.dining,matNeg:i.cream,openings:[{a:-.5,b:3,bottom:0,top:7}]}),n.wall({...r,axis:"z",at:-6,from:-14.7,to:-6,matPos:i.kitchen,matNeg:i.br2}),n.wall({...r,axis:"x",at:-5,from:-6,to:6,matPos:i.dining,matNeg:i.kitchen,openings:[{a:-2,b:2,bottom:0,top:7}]}),n.wall({...r,axis:"z",at:6,from:-14.7,to:-5,matPos:e.siding,matNeg:i.kitchen,openings:[a(-12,-9.4)]}),n.wall({...r,axis:"z",at:6,from:-5,to:-3,matPos:e.siding,matNeg:i.dining}),n.wall({...r,axis:"z",at:6,from:-3,to:5,matPos:i.cream,matNeg:i.dining,openings:[{a:-2.5,b:1.1,bottom:0,top:7}]}),n.wall({...r,axis:"z",at:6,from:5,to:14.7,matPos:i.cream,matNeg:i.cream,openings:[{a:12,b:14.5,bottom:0,top:7}]}),n.wall({...r,t:.5,axis:"x",at:-3,from:6,to:18.7,matPos:i.cream,matNeg:e.siding,openings:[o(8.6,11.4,3,7),o(13.4,16.4,3,7)]}),n.wall({...r,axis:"x",at:5,from:-6,to:3,matPos:i.cream,matNeg:i.dining,openings:[{a:-3.6,b:-.4,bottom:0,top:7}]}),n.box(-1.5,-1.35,0,6.8,11.8,14.7,e.trim,{collide:!0}),n.box(-1.62,-1.5,3.1,3.4,12,12.2,K("#c9a44f",{metalness:.8,roughness:.3}),{occlude:!1});let h=new Xe([new j(-He,0),new j(-be,0),new j(-be,8)]),u=new Pt(new jn(h,{depth:3,bevelEnabled:!1}),e.trim);u.rotation.y=Math.PI/2,u.position.set(3,0,0),n.add(u);let d=[e.trim,e.trim,e.oak,e.trim,e.trim,e.trim];for(let P=0;P<12;P++){let k=He-P*ps;n.box(3.05,5.95,P*Zs,(P+1)*Zs,k-ps,k+(P===0?0:.08),d,{occlude:!1})}n.surface(3,6,be,He,(P,k)=>__(k)),n.collider(2.85,3.15,0,17,be,He-.3),n.collider(3,6,0,7,be-.25,be+.1);let f=K("#a8692f",{roughness:.5});for(let P=0;P<8;P++){let k=He-(P+.5)*ps,nt=(P+1)*Zs;n.box(3,3.14,nt,nt+3+0,k-.07,k+.07,f,{occlude:!1})}n.box(2.9,3.3,0,4,He-.45,He-.05,f,{occlude:!1}),n.slopeZ(He-.25,3.6,He-8*ps,8*Zs+3.2,2.95,3.2,.18,f,{occlude:!1}),n.box(3.4,5.8,0,6.8,be-.12,be-.02,e.trim,{occlude:!1}),n.box(5.3,5.5,3.1,3.3,be-.22,be-.1,K("#c9a44f",{metalness:.8,roughness:.3}),{occlude:!1}),n.box(-5.8,-5.75,0,6.8,11.2,13.8,K("#f2f1ec"),{occlude:!1});for(let P=.6;P<6.6;P+=.3)n.box(-5.75,-5.7,P,P+.12,11.3,13.7,e.trim,{occlude:!1});n.box(-18.7,6,-.3,.01,-14.7,14.7,e.oak,{occlude:!1,cast:!1}),n.box(6,18.7,-.3,.01,-2.75,14.7,e.oak,{occlude:!1,cast:!1}),n.box(-5.8,5.8,.01,.025,-14.7,-5.2,e.kitchenTile,{occlude:!1,cast:!1}),n.box(-18.7,-9.2,.01,.025,1.2,5.8,e.bathTile,{occlude:!1,cast:!1}),n.box(-18.7,-6.2,.01,.025,-14.7,-6.2,e.beigeCarpet,{occlude:!1,cast:!1}),n.box(6.3,18.7,-1.02,-.99,-14.7,-3.25,e.blueCarpet,{occlude:!1,cast:!1}),n.surface(-19,6,-15,15,0),n.surface(6,19,-3,15,0),n.surface(6,19,-15,-3,-1);let g=[[-19.3,3,-9,15.3],[6,19.3,-9,15.3],[3,6,-9,be],[3,6,He,15.3]];for(let[P,k,nt,tt]of g)n.box(P,k,8,8.5,nt,tt,e.ceiling),n.box(Math.max(P,-18.7),Math.min(k,18.7),8.5,8.52,nt,Math.min(tt,11),e.oak,{occlude:!1,cast:!1});n.box(-19.3,19.3,8,8.4,-15.3,-9,e.ceiling),n.box(5.8,6.2,8,8.5,be,He,e.trim),n.surface(-19,3,-9,11,8.5),n.surface(6,19,-9,11,8.5),n.surface(3,6,-9,be,8.5),n.box(-1.5,1.5,7.9,7.98,-12.2,-10.2,K("#ffffff",{emissive:"#e8f4ff",emissiveIntensity:1.4}),{occlude:!1,cast:!1});let _={y0:8.5,y1:17.6,t:.6,trim:e.trim,glass:e.glass,edge:e.trim};n.wall({..._,axis:"x",at:-9,from:-19.3,to:-1,matPos:i.upW,matNeg:e.siding,openings:[c(o(-12.5,-9.5,2.5,6))]}),n.wall({..._,axis:"x",at:-9,from:-1,to:6,matPos:i.bath2,matNeg:e.siding,openings:[c(o(1.2,3.6,3.5,6))]}),n.wall({..._,axis:"x",at:-9,from:6,to:19.3,matPos:i.upE,matNeg:e.siding,openings:[c(o(11,14,2.5,6))]});for(let[P,k]of[[-12.5,-9.5],[1.2,3.6],[11,14]])n.slopeZ(-9.3,15.3,-10.8,14.3,P-.3,k+.3,.08,e.trim,{occlude:!1});let m={y0:8.5,y1:16,t:.4,trim:e.trim,glass:e.glass};n.wall({...m,y1:12.5,axis:"x",at:11,from:-18.7,to:-1,matPos:i.upW,matNeg:i.upW}),n.wall({...m,y1:12.5,axis:"x",at:11,from:-1,to:3,matPos:i.landing,matNeg:i.landing}),n.wall({...m,y1:12.5,axis:"x",at:11,from:6,to:18.7,matPos:i.upE,matNeg:i.upE}),n.wall({...m,axis:"z",at:-1,from:-8.7,to:-3,matPos:i.bath2,matNeg:i.upW}),n.wall({...m,axis:"z",at:-1,from:-3,to:6.8,matPos:i.landing,matNeg:i.upW,openings:[c(a(-2.2,.4))]}),n.wall({...m,axis:"z",at:6,from:-8.7,to:-3,matPos:i.upE,matNeg:i.bath2}),n.wall({...m,axis:"z",at:6,from:-3,to:6.8,matPos:i.upE,matNeg:i.landing,openings:[c(a(-2.2,.4))]}),n.wall({...m,axis:"x",at:-3,from:-1,to:6,matPos:i.landing,matNeg:i.bath2,openings:[c(a(0,2.5))]}),n.wall({...m,axis:"z",at:3,from:be,to:6.8,matPos:i.landing,matNeg:i.landing});let p=new Xe([[6.8,8.5],[11,8.5],[11,12.5],[6.8,16]].map(([P,k])=>new j(-P,k)));for(let[P,k]of[[-1,i.upW],[3,i.landing],[6,i.upE]]){let nt=new Pt(new jn(p,{depth:.4,bevelEnabled:!1}),k);nt.rotation.y=Math.PI/2,nt.position.x=P-.2,n.add(nt),n.collider(P-.2,P+.2,8.5,16,6.8,11)}n.wall({...m,axis:"x",at:be,from:-1,to:3,matPos:i.landing,matNeg:i.landing}),n.box(-.4,2.4,8.5,15.3,be-.3,be-.2,e.trim,{occlude:!1}),n.slopeZ(11,12.5,6.8,16,-18.7,18.7,.15,e.ceiling),n.box(-18.7,18.7,16,16.3,-8.7,6.8,e.ceiling);let S=[[15.3,8.5],[15.3,9],[3,19.25],[-9.3,17.6],[-9.3,11.5],[-15.3,9],[-15.3,8.5]],M=(P,k,nt)=>{let tt=new Xe(S.map(([Z,st])=>new j(-Z,st)));for(let[Z,st,A,St]of k)tt.holes.push(new Ei([new j(-Z,A),new j(-st,A),new j(-st,St),new j(-Z,St)]));let wt=new jn(tt,{depth:.6,bevelEnabled:!1}),Dt=new Pt(wt,[e.siding,e.trim]);Dt.rotation.y=Math.PI/2,Dt.position.x=P>0?18.7:-19.3,n.add(Dt);let Lt=new Xe([[11,8.5],[11,12.5],[6.8,16],[-8.7,16],[-8.7,8.5]].map(([Z,st])=>new j(-Z,st)));Lt.holes=tt.holes;let Yt=new Pt(new wi(Lt),nt.clone());Yt.material.side=ve,Yt.rotation.y=Math.PI/2,Yt.position.x=P>0?18.68:-18.68,n.add(Yt);for(let[Z,st,A,St]of k)n.openingTrim("z",P*19,Math.min(Z,st),Math.max(Z,st),A,St,.6,e.trim,!0),n.windowGlass("z",P*19,Math.min(Z,st),Math.max(Z,st),A,St,e.glass,e.trim);n.box(P*19-.35,P*19+.35,16.6,18.2,2.3,3.7,e.trim,{occlude:!1}),n.collider(P*19-.3,P*19+.3,8.5,17,-9,11)};M(-1,[[0,3,10.5,14.5]],i.upW),M(1,[[-4,-1.5,10.5,14.5]],i.upE),n.slopeZ(16.3,8.17,3,19.25,-20,20,.35,e.roof),n.slopeZ(3,19.25,-9.9,17.5,-20,20,.35,e.roof),n.slopeZ(-9.3,11.5,-16.3,8.58,-20,20,.35,e.roof),n.box(-20.1,20.1,19.3,19.7,2.7,3.5,e.roof),n.box(-20,20,7.75,8.4,16.2,16.45,e.trim,{occlude:!1}),n.box(-20,20,8.15,8.75,-16.45,-16.2,e.trim,{occlude:!1}),n.box(-20,20,16.9,17.6,-10.1,-9.85,e.trim,{occlude:!1});for(let P of[-1,1]){let k=P>0?19.3:-19.65,nt=P>0?19.65:-19.3;n.slopeZ(16.3,7.9,3,18.95,k,nt,.55,e.trim,{occlude:!1}),n.slopeZ(3,18.95,-9.9,17.2,k,nt,.55,e.trim,{occlude:!1}),n.slopeZ(-9.3,11.2,-16.3,8.3,k,nt,.55,e.trim,{occlude:!1})}let x=9+(-12+15.3)*(2.5/6);n.slopeZ(-11,x+.85,-13,x+.02,-1.4,1.4,.3,K("#9fbdd2",{roughness:.1,metalness:.3}),{occlude:!1}),n.box(19.3,22.3,-2.5,7.5,3.6,7.4,e.brick,{collide:!0}),n.box(19.3,21.6,7.5,22,4.3,6.7,e.brick,{collide:!0}),n.box(19.1,21.8,22,22.4,4.1,6.9,e.concrete);let D=[e.brick,e.brick,e.porchFloor,e.brick,e.brick,e.brick];n.box(-4.5,4.5,-2.5,0,15.3,20.8,D,{collide:!1}),n.surface(-4.5,4.5,15,20.8,0),n.collider(-4.6,-4.4,-2.5,.2,15.3,20.8),n.collider(4.4,4.6,-2.5,.2,15.3,20.8),n.collider(-4.6,-3.6,-2.5,.2,20.6,20.9),n.collider(3.6,4.6,-2.5,.2,20.6,20.9);let T=[e.brick,e.brick,K("#8b6f5a",{roughness:.8}),e.brick,e.brick,e.brick];for(let P=0;P<4;P++){let k=-.5*(P+1),nt=20.8+P,tt=20.8+P+1;n.box(-3.6,3.6,-2.5,k,nt,tt,T,{occlude:!1}),n.surface(-3.6,3.6,nt,tt,k)}for(let P of[-1,1]){n.railing(P*4.3,15.6,P*4.3,20.4,0,2.8,e.trim,{spacing:.4}),n.box(P*4.3-.25,P*4.3+.25,0,7.9,20.15,20.65,e.trim,{collide:!0}),n.box(P*4.3-.2,P*4.3+.2,0,7.9,15.3,15.6,e.trim,{occlude:!1});let k=P*3.8;n.box(k-.2,k+.2,-2.5,.8,24.4,24.8,e.trim,{collide:!0}),n.slopeZ(20.6,2.8,24.6,.8,k-.11,k+.11,.14,e.trim,{occlude:!1});for(let nt=0;nt<4;nt++){let tt=21.3+nt,wt=2.8-(tt-20.6)/4*2;n.box(k-.05,k+.05,-.5*(nt+1),wt,tt-.05,tt+.05,e.trim,{occlude:!1})}n.collider(k-.15,k+.15,-2.5,3,20.8,24.8)}n.box(-4.7,4.7,7.6,8.4,20.2,20.8,e.trim),n.box(-4.6,4.6,7.9,8.2,15.3,20.8,e.ceiling),n.slopeX(-5.3,8.3,0,11.7,15.3,21.3,.3,e.roof),n.slopeX(0,11.7,5.3,8.3,15.3,21.3,.3,e.roof);let R=new Pt(new wi(new Xe([new j(-4.9,8.4),new j(4.9,8.4),new j(0,11.5)])),new pe({map:e.clapboard.map,side:ve,roughness:.8}));R.position.z=20.85,n.add(R);let C=new pe({map:lo("#1f3557","#f2f2ec",16),roughness:.9,side:ve});C.map.repeat.set(1/8.4,1),n.slopeZ(20.7,8,21.9,7,-4.2,4.2,.05,C,{occlude:!1,cast:!0}),n.box(-4.2,4.2,6.5,7.02,21.86,21.92,C,{occlude:!1});for(let P of[-2.15,2.15])n.box(P-.2,P+.2,5,5.8,15.3,15.6,K("#1b1b1b",{emissive:"#ffcf80",emissiveIntensity:.5}),{occlude:!1});let b=new Pt(new Un(1.4,.5),new pe({map:zc("419"),roughness:.6}));b.position.set(0,-.25,20.81),n.add(b,{occlude:!1});let y=new Ot,I=K("#1c1c1c",{roughness:.5,metalness:.6}),B=new Pt(new ge(1.6,.1,3.4),I);B.position.y=1.5,y.add(B);let N=new Pt(new ge(.1,1.4,3.4),I);N.position.set(-.75,2.2,0),y.add(N);for(let P of[-1.6,1.6]){let k=new Pt(new ge(1.4,1.5,.1),I);k.position.set(0,.75,P),y.add(k)}y.position.set(-3.2,0,18),n.add(y,{occlude:!1}),n.colliderFromObject(y),n.wall({axis:"z",at:19,from:-14.7,to:-3,y0:-1,y1:1.4,t:.6,matPos:e.siding,matNeg:i.porch,edge:e.trim}),n.wall({axis:"x",at:-15,from:6,to:19.3,y0:-1,y1:1.4,t:.6,matPos:i.porch,matNeg:e.siding,edge:e.trim,openings:[{a:7,b:9.8,bottom:-1,top:6.4}]}),n.box(18.7,19.3,7.3,8.5,-14.7,-3,e.trim),n.box(6,19.3,7.3,8.5,-15.3,-14.7,e.trim),n.box(18.97,19.03,1.4,7.3,-14.7,-3,e.screen,{cast:!1}),n.box(9.8,18.7,1.4,7.3,-15.03,-14.97,e.screen,{cast:!1}),n.box(6,7,1.4,7.3,-15.03,-14.97,e.screen,{cast:!1}),n.box(7,9.8,6.4,7.3,-15.03,-14.97,e.screen,{cast:!1}),n.collider(18.7,19.3,-1,8.5,-15.3,-3),n.collider(9.8,19.3,-1,8.5,-15.3,-14.7),n.collider(6,7,-1,8.5,-15.3,-14.7);for(let P=-14.7;P<=-3;P+=2.9)n.box(18.75,19.25,1.4,7.3,P-.2,P+.2,e.trim,{occlude:!1});for(let P of[6.5,7,9.8,12.8,15.8,18.9])n.box(P-.2,P+.2,1.4,7.3,-15.25,-14.75,e.trim,{occlude:!1});n.box(6.6,9.8,6.4,6.8,-15.25,-14.75,e.trim,{occlude:!1}),n.box(7,9.8,-2.5,-1.75,-16.5,-15.3,e.concrete,{occlude:!1}),n.surface(7,9.8,-16.5,-15.3,-1.75);let H=K("#9a4b3a",{roughness:.7});n.box(16.5,18.7,0,.07,3.7,7.3,H,{occlude:!1}),n.box(18.45,18.7,0,3.3,4.2,6.8,e.brick,{occlude:!1}),n.box(18.38,18.46,.15,2.85,4.55,6.45,K("#b8933f",{metalness:.8,roughness:.35}),{occlude:!1}),n.box(18.35,18.4,.3,2.7,4.7,6.3,K("#0d0d0d",{roughness:.15}),{occlude:!1}),n.box(18.2,18.7,0,3.7,3.4,4.2,e.trim,{collide:!0}),n.box(18.2,18.7,0,3.7,6.8,7.6,e.trim,{collide:!0}),n.box(18.2,18.7,3.3,3.85,3.4,7.6,e.trim),n.box(17.85,18.7,3.85,4.05,3.15,7.85,e.trim),Wc(n,18.6,4.6,5.5,-Math.PI/2,{w:4.2,h:2.4}),kc(n,7.9,7.6,Math.PI/2,{w:7.5,color:"#26416b"}),uo(n,7.4,2.6,0,{w:1.8,d:1.8,h:2.2,top:"#7a4f2e"}),Ys(n,7.4,2.2,2.6,{base:"#6c9ab6"}),iu(n,9.8,.6,Math.PI/2+.5,{color:"#8e9398"});let Y=lo("#2b3f63","#efe9db",14);Y.repeat.set(.6,.6),qs(n,10.3,13.2,Math.PI,{stripe:Y}),qs(n,15.2,13.2,Math.PI,{stripe:Y}),Gc(n,12.75,13.6,{r:.75,h:2.2}),su(n,17.4,12.3);let V=K("#a68457",{roughness:1});n.box(17.2,18.4,0,1.4,1.6,2.9,V,{collide:!0,occlude:!1}),n.box(17.2,18.4,0,1.4,8.1,9.4,V,{collide:!0,occlude:!1}),Gc(n,11,-.3,{r:1.5,h:3.3,top:"#2d3540"}),fo(n,9.8,-1.4),fo(n,12.3,-1.2),fo(n,11.2,1.1),ou(n,17.9,-1.4,-Math.PI/2),ii(n,"z",6.2,7.6,6.6,3.4,2.3,1,ni(3)),n.box(15.6,16.6,4.6,6,14.65,14.7,K("#9a6a3d"),{occlude:!1}),ru(n,-.6,-1.4,0),Fe(n,-.6,8,-1.4,{chandelier:!0}),ii(n,"x",-4.8,-4,6,2.4,1.8,1,ni(8)),ii(n,"x",5,1.6,6.2,1.8,1.4,-1,ni(5));let rt=(P,k,nt,tt)=>{n.box(P,k,0,2.9,nt,tt,e.maple,{collide:!0,occlude:!1}),n.box(P-.05,k+.05,2.9,3.05,nt-.05,tt+.05,e.counter,{occlude:!1})};rt(-5.8,4.4,-14.7,-12.7),n.box(-.8,.8,3.03,3.07,-14.3,-13.1,K("#ffffff",{roughness:.2}),{occlude:!1}),n.box(-.08,.08,3.05,4,-14.5,-14.35,K("#c9ccd0",{metalness:.9,roughness:.2}),{occlude:!1}),n.box(1,3.4,.15,2.85,-12.72,-12.62,K("#fafaf8",{roughness:.3}),{occlude:!1});for(let P=-5.6;P<.9;P+=1.6)n.box(P,P+1.4,.4,2.6,-12.72,-12.66,K("#e4d2b4"),{occlude:!1});n.box(-5.8,-2.3,4.7,7.6,-14.7,-13.5,e.maple,{occlude:!1}),n.box(2.3,4.6,4.7,7.6,-14.7,-13.5,e.maple,{occlude:!1}),rt(-5.8,-3.8,-12.7,-10.6),rt(-5.8,-3.8,-8,-6.3),cu(n,-4.75,-9.3,Math.PI/2),n.box(-5.8,-4.6,4.7,7.6,-12.7,-10.6,e.maple,{occlude:!1}),n.box(-5.8,-4.6,4.7,7.6,-8,-6.3,e.maple,{occlude:!1}),n.box(-5.8,-4,5.6,6.6,-10.6,-8,K("#f2f2f0"),{occlude:!1}),au(n,4.55,-7,-Math.PI/2),n.box(4.4,5.2,3.05,4,-14.3,-13.8,K("#1c1c1c"),{occlude:!1}),Fe(n,0,8,-8.2),Vc(n,17,-8.8,-Math.PI/2,{w:6.8,y:-1}),Vc(n,12.6,-4.9,Math.PI,{w:5.2,y:-1,cushion:"#2a4a7c"});let X=lo("#1f3557","#f4f4f0",10);ho(n,11.6,-12.9,.2,{y:-1,stripe:X}),ho(n,15.2,-13,-.2,{y:-1}),ho(n,8.4,-8.5,Math.PI/2,{y:-1}),uo(n,13.2,-8.8,0,{w:3.2,d:2,h:1.6,y:-1,top:"#e3d6bb"}),kc(n,-12.5,13.1,Math.PI,{w:7,color:"#6f7d8c"}),an(n,-12.5,7.1,0,{w:4.8,h:2.2,d:1.5,color:"#3c3a38"}),Wc(n,-12.5,2.45,7.1,0,{w:4.4,h:2.5}),qs(n,-17,9.5,Math.PI/2,{color:"#c8b99a"}),po(n,-16,-9,8.6,11.8,.012,K("#b7a58a",{roughness:1})),Fe(n,-12.5,8,10.5),qc(n,-18.6,-16.1,1.3,5.8),n.box(-16.08,-16.02,1.8,7.2,1.3,5.8,K("#ffffff",{transparent:!0,opacity:.85}),{occlude:!1}),Xc(n,-14.4,5,Math.PI),Yc(n,-12,2.15,0,{w:2.8}),ds(n,-15.3,-2.6,Math.PI/2,{w:3.4,l:6.6,pillows:1,quilt:us(),frame:"#f0ede4"}),an(n,-11.2,-5,0,{w:3.6,h:3.4,color:"#e9e4d8"}),Fe(n,-14,8,-2.5),ii(n,"x",.8,-15.5,5.6,2,1.4,-1,ni(11)),ds(n,-13,-9.8,Math.PI,{w:5,l:6.8,quilt:us("#efe6d6")}),an(n,-16.4,-6.95,0,{w:1.6,h:2.1,d:1.4}),an(n,-9.6,-6.95,0,{w:1.6,h:2.1,d:1.4}),Ys(n,-16.4,2.3,-6.95,{base:"#8a6a4a"}),Ys(n,-9.6,2.3,-6.95,{base:"#8a6a4a"}),an(n,-7,-11.8,-Math.PI/2,{w:4.2,h:4,d:1.6,color:"#6b4428"}),uo(n,-10.2,-13.9,0,{w:3.6,d:1.6,h:2.5,top:"#7a4f2e"}),Fe(n,-12.5,8,-10.5),ii(n,"x",-6,-13,6.3,1.4,1.1,-1,ni(17)),Fe(n,-7.5,8,0),Fe(n,0,8,10),Fe(n,12.5,8,6),Fe(n,12.5,8,-9),Fe(n,-14,8,3.5,{r:.4});let it=8.5;ds(n,-15.3,-4.6,Math.PI/2,{w:3.4,l:6.6,pillows:1,y:it,quilt:us("#f1e9dc"),frame:"#efece4"}),ds(n,-15.3,6,Math.PI/2,{w:3.4,l:6.6,pillows:1,y:it,quilt:us("#dfe8f0"),frame:"#efece4"}),an(n,-5.2,-7.9,0,{w:3.6,h:3.6,y:it,color:"#dad6cc"}),an(n,-2.1,3.5,-Math.PI/2,{w:3,h:4.2,y:it,color:"#c8c6be"}),po(n,-14,-6,-2,4,it+.03,K("#e8dfcc",{roughness:1}));let dt=new Ot,vt=K("#f1efe9"),zt=new Pt(new fn(.35,.35,.5,12),vt);dt.add(zt);for(let P=0;P<5;P++){let k=new Pt(new ge(2,.05,.5),vt);k.position.x=1.2;let nt=new Ot;nt.rotation.y=P/5*Math.PI*2,nt.add(k),dt.add(nt)}dt.position.set(-10,15.4,.5),n.add(dt,{occlude:!1}),ii(n,"z",-1.2,6,it+5.6,2,1.5,-1,ni(23)),ds(n,15.1,3.4,-Math.PI/2,{w:5,l:6.8,y:it,quilt:us("#ece2d2")}),an(n,17.9,-.4,-Math.PI/2,{w:1.6,h:2.1,d:1.4,y:it}),an(n,17.9,7.2,-Math.PI/2,{w:1.6,h:2.1,d:1.4,y:it}),Ys(n,17.9,it+2.3,7.2,{base:"#7a96ad"}),an(n,8.6,-7.9,0,{w:4,h:3.4,y:it}),qs(n,8.5,6.5,Math.PI/2+.4,{color:"#c3cdd6",y:it}),po(n,10,17,0,7,it+.03,K("#9fb2c6",{roughness:1})),ii(n,"z",6.2,3.2,it+5.6,2.6,1.8,1,ni(29)),n.box(-.6,5.6,it+.02,it+.035,-8.6,-3.2,e.bathTile,{occlude:!1,cast:!1}),qc(n,-.8,4,-8.6,-6.2,{y:it}),Xc(n,5,-7.4,-Math.PI/2,{y:it}),Yc(n,4.9,-4.6,-Math.PI/2,{w:2.4,y:it}),Fe(n,-10,16,-4,{r:.5}),Fe(n,12.5,16,-2),Fe(n,2.2,16,-1),Fe(n,2.5,16,-6,{r:.4});let Kt=[[12.5,6.8,7,30],[12.5,6.8,-.5,18],[-.6,5.8,-1.4,24],[0,7.2,-9.5,24],[0,7.2,10,20],[12.5,6.5,-9,22],[-12.5,7.2,10.5,20],[-12.5,7.2,-10.5,20],[-13,7.2,-1.5,18],[-10,15,0,28],[12,15,0,28],[2.2,15,-3,14]];for(let[P,k,nt,tt]of Kt){let wt=new Qr("#ffe9c8",tt,26,1.5);wt.position.set(P,k,nt),n.scene.add(wt)}x_(n,e)}function x_(n,t){let i=new pe({map:Qh(),roughness:1});n.box(-400,400,-2.5-.2,-2.5,-400,400,i,{occlude:!1,cast:!1});let s=new pe({map:tu(),roughness:.95});n.box(-400,400,-2.5,-2.5+.02,44,74,s,{occlude:!1,cast:!1}),n.box(-400,400,-2.5,-2.5+.4,43.4,44,t.concrete,{occlude:!1}),n.box(-400,400,-2.5,-2.5+.06,38,43,t.concrete,{occlude:!1,cast:!1}),n.box(-400,400,-2.5,-2.5+.06,76,81,t.concrete,{occlude:!1,cast:!1}),n.box(-400,400,-2.5,-2.5+.4,74,74.6,t.concrete,{occlude:!1}),n.box(-1.6,1.6,-2.5,-2.5+.05,24.8,38,t.concrete,{occlude:!1,cast:!1});let r=new pe({map:eu(),roughness:1});n.box(-25,-20.5,-2.5,-2.5+.04,-110,43.4,r,{occlude:!1,cast:!1}),n.box(-25,25,-2.5,-2.5+.04,-125,-110,r,{occlude:!1,cast:!1}),n.box(-25,-19.3,-2.5,-2.5+.03,-22,-15.3,r,{occlude:!1,cast:!1});let o=new pe({map:nu(),roughness:.95});n.box(-19.3,19.3,-2.5,-2.5+.05,-26,-15.3,o,{occlude:!1,cast:!1});let a=K("#f4f4f1",{roughness:.5});n.box(-9.1,-8.8,-2.5,4.3,-18.8,-15.3,a,{collide:!0}),n.box(-4.7,-4.4,-2.5,4.3,-18.8,-15.3,a,{collide:!0}),n.box(-8.8,-7.1,-2.5,4.3,-18.8,-18.5,a,{collide:!0}),n.box(-4.9,-4.7,-2.5,4.3,-18.8,-18.5,a,{collide:!0}),n.box(-7.1,-6.95,-2.5+.2,4,-21.2,-18.8,a,{collide:!0}),n.box(-8.8,-4.7,-2.5,-2.5+.25,-18.5,-15.3,K("#8a6a4a"),{occlude:!1}),n.box(-6.85,-6.65,-2.5+1,4.6,-15.45,-15.3,K("#c9ccd0",{metalness:.9,roughness:.2}),{occlude:!1}),n.box(-7.1,-6.4,4.3,4.45,-16.3,-15.3,K("#c9ccd0",{metalness:.9,roughness:.2}),{occlude:!1}),n.slopeZ(-15.3,-1.3,-19,-2.35,-17.5,-13.5,.2,K("#f0f0ec",{roughness:.5})),n.collider(-17.5,-13.5,-2.5,-1.25,-19,-15.3),n.box(15.8,18.6,-2.5,-2.5+2.8,-18.4,-15.8,K("#c9c6bd",{roughness:.6}),{collide:!0}),n.box(16.2,18.2,-2.5+2.8,-2.5+2.85,-18,-16.2,K("#2a2a2a"),{occlude:!1}),lu(n,11,-21,.15),hu(n,16,-100),mo(n,-27.5,-25.5,-110,30,8),mo(n,24.5,26.5,-110,-22,9),mo(n,-25.5,24.5,-112,-110,7);for(let h=7;h<=18;h+=2.6)fs(n,h,16.6,{r:1.5,color:"#3d6b2b",flowers:h%2<1.3?"#d98fb4":"#8fb0de"});for(let h of[-18,-16,-11,-5.5])fs(n,h,16.4,{r:1.2,color:"#4a7a35"});fs(n,-7.8,23.8,{r:3.2,color:"#3b6b2a"}),fs(n,22.5,-4,{r:2,color:"#3b6b2a"}),fs(n,22,10,{r:1.8,color:"#3f6e2c",flowers:"#d98fb4"}),Nn(n,-8.5,25,{trunk:5,crown:9,maple:!0,seed:5,lean:.08}),Nn(n,18,-48,{trunk:12,crown:14,seed:3}),Nn(n,-12,-72,{trunk:14,crown:15,color:"#466f2f",seed:9}),Nn(n,8,-88,{trunk:10,crown:11,seed:13}),Nn(n,-18,-36,{trunk:8,crown:9,color:"#527f36",seed:21}),Nn(n,30,30,{trunk:10,crown:12,seed:17}),Nn(n,-40,34,{trunk:11,crown:12,seed:27,maple:!0}),pn(n,-52,0,{w:26,d:40,h:20,color:"#e8ecee"}),pn(n,50,-2,{w:24,d:36,h:19,color:"#f1f0ea",roof:"#55595f"}),pn(n,-100,4,{w:28,d:38,h:22,color:"#a8bed1"}),pn(n,100,0,{w:26,d:34,h:18,color:"#e7dfcc"}),pn(n,-45,110,{w:28,d:34,h:21,color:"#dbe3e8"}),pn(n,5,112,{w:26,d:36,h:18,color:"#f3f2ec"}),pn(n,55,110,{w:30,d:36,h:22,color:"#9fb6c9"}),pn(n,-10,-160,{w:28,d:36,h:20,color:"#ece9df"}),pn(n,45,-150,{w:24,d:30,h:17,color:"#d9e2e6"});for(let h of[-80,-30,30,80])Nn(n,h,41,{trunk:12,crown:11,seed:Math.abs(h)+3});let c=K("#2f3a2f",{metalness:.4,roughness:.6});n.box(-30.15,-29.85,-2.5,-2.5+9,42.2,42.5,c,{collide:!0});let l=new Pt(new Un(5,1.25),new pe({map:zc("Jersey Ave"),side:ve}));l.position.set(-30,-2.5+8.2,42.35),n.add(l,{occlude:!1}),n.collider(-400,400,-2.5,30,120,122),n.collider(-400,400,-2.5,30,-200,-198),n.collider(-200,-198,-2.5,30,-200,122),n.collider(198,200,-2.5,30,-200,122)}var y_=[{name:"Living Room",level:0,box:[6,19,-3,15]},{name:"Foyer",level:0,box:[-6,6,5,15]},{name:"Dining Room",level:0,box:[-6,6,-5,5]},{name:"Kitchen",level:0,box:[-6,6,-15,-5]},{name:"Screened Porch",level:0,box:[6,19,-15,-3]},{name:"Front Bedroom (TV Room)",level:0,box:[-19,-6,6,15]},{name:"Hall Bathroom",level:0,box:[-19,-9,1,6]},{name:"Hallway",level:0,box:[-9,-6,-6,6]},{name:"Middle Bedroom",level:0,box:[-19,-9,-6,1]},{name:"Back Bedroom",level:0,box:[-19,-6,-15,-6]},{name:"Front Porch",level:0,box:[-4.6,4.6,15,21]},{name:"Upstairs Bedroom (Left)",level:1,box:[-19,-1,-9,11]},{name:"Upstairs Bedroom (Right)",level:1,box:[6,19,-9,11]},{name:"Upstairs Bathroom",level:1,box:[-1,6,-9,-3]},{name:"Upstairs Landing",level:1,box:[-1,6,-3,11]}];function du(n,t,e){if(n>3&&n<6&&e>be&&e<He&&t>.3&&t<8.3)return"Staircase";let i=t>4?1:0;for(let s of y_){let[r,o,a,c]=s.box;if(s.level===i&&n>=r&&n<=o&&e>=a&&e<=c)return s.name}return e>43.4&&e<74?"Jersey Avenue":e>=21&&e<=43.4&&n>-26&&n<26?"Front Yard":e<-15&&e>-112&&n>-26&&n<26?e<-95?"Back of the Yard":"Backyard":n>-26&&n<26&&e>-112&&e<43?n<-19?"Driveway":"Side Yard":"Spring Lake"}var Zc=[{name:"Front walk",pos:[0,-2.5,34],yaw:0},{name:"Front door",pos:[0,0,12.5],yaw:0},{name:"Living room",pos:[12.5,0,4],yaw:.3},{name:"Dining room",pos:[-3,0,3.2],yaw:0},{name:"Kitchen",pos:[0,0,-7.5],yaw:0},{name:"Screened porch",pos:[10.5,-1,-10.5],yaw:-.8},{name:"Bedrooms",pos:[-7.5,0,3],yaw:0},{name:"Upstairs",pos:[2.5,8.5,-1],yaw:Math.PI/2},{name:"Backyard",pos:[2,-2.5,-38],yaw:Math.PI}];function fu({shirt:n="#3d6fa8",pants:t="#2f3540",skin:e="#e2b895",hair:i="#4a3322",shoes:s="#f2f2f2"}={}){let r=new Ot,o=(B,N=.75)=>new pe({color:B,roughness:N}),a=o(n),c=o(t),l=o(e,.6),h=o(i,.9),u=o(s,.5),d=new Ot;r.add(d);let f=new Pt(new ss(.62,1.15,6,14),a);f.scale.set(1,1,.62),f.position.y=3.95,d.add(f);let g=new Pt(new ss(.55,.2,4,12),c);g.scale.set(1.05,1,.65),g.position.y=3.05,d.add(g);let _=new Pt(new fn(.16,.18,.35,10),l);_.position.y=5,d.add(_);let m=new Ot;m.position.y=5.48,d.add(m);let p=new Pt(new ti(.42,20,16),l);p.scale.set(.92,1.05,1),m.add(p);let S=new Pt(new ti(.45,20,12,0,Math.PI*2,0,Math.PI*.55),h);S.position.set(0,.05,-.04),S.scale.set(.95,1,1.02),m.add(S);let M=o("#20252b",.3);for(let B of[-.14,.14]){let N=new Pt(new ti(.045,8,8),M);N.position.set(B,.04,.39),m.add(N)}let x=new Pt(new ti(.06,8,8),l);x.position.set(0,-.05,.42),m.add(x);let D=(B,N,H)=>{let Y=new Ot,V=new Pt(new ss(N,B,4,10),H);return V.position.y=-B/2-N*.3,Y.add(V),Y},T=[];for(let B of[-.3,.3]){let N=D(1.25,.21,c);N.position.set(B,3,0);let H=D(1.15,.18,c);H.position.y=-1.42,N.add(H);let Y=new Pt(new ge(.36,.22,.72),u);Y.position.set(0,-1.38,.14),H.add(Y),d.add(N),T.push({hip:N,knee:H})}let R=[];for(let B of[-.78,.78]){let N=D(.95,.15,a);N.position.set(B,4.62,0);let H=D(.85,.13,l);H.position.y=-1.08,N.add(H);let Y=new Pt(new ti(.15,10,8),l);Y.position.y=-1.05,H.add(Y),d.add(N),R.push({shoulder:N,elbow:H,side:Math.sign(B)})}r.traverse(B=>{B.isMesh&&(B.castShadow=!0,B.receiveShadow=!0)});let C=0,b=0;r.userData.animate=(B,N)=>{let H=Math.min(1,N/6);b+=(H-b)*Math.min(1,B*8),C+=B*(3.2+N*.75);let Y=Math.sin(C),V=.55*b;T[0].hip.rotation.x=Y*V,T[1].hip.rotation.x=-Y*V,T[0].knee.rotation.x=Math.max(0,-Math.cos(C))*V*1.3,T[1].knee.rotation.x=Math.max(0,Math.cos(C))*V*1.3;for(let X of R){let it=X.side<0?-1:1;X.shoulder.rotation.x=Y*V*.8*it,X.shoulder.rotation.z=X.side*.08,X.elbow.rotation.x=-.25-b*.2}d.position.y=Math.abs(Math.cos(C))*.12*b,f.rotation.y=Y*.06*b;let rt=Math.sin(performance.now()*.0015)*.02*(1-b);f.scale.y=1+rt};let y=[a,c,l,h,u,M],I=1;return r.userData.setOpacity=B=>{if(!(Math.abs(B-I)<.01)){I=B;for(let N of y)N.transparent=B<.99,N.opacity=B,N.depthWrite=B>=.99}},r.userData.setColors=({shirt:B,pants:N})=>{B&&a.color.set(B),N&&c.color.set(N)},r}var si=.75,v_=5.9,M_=7,S_=13,E_=40,b_=document.getElementById("app"),Ke=new kr({antialias:!0,powerPreference:"high-performance",preserveDrawingBuffer:!0});Ke.setPixelRatio(Math.min(window.devicePixelRatio,2));Ke.setSize(window.innerWidth,window.innerHeight);Ke.shadowMap.enabled=!0;Ke.shadowMap.type=bc;Ke.toneMapping=wc;Ke.toneMappingExposure=1.05;Ke.outputColorSpace=xe;b_.appendChild(Ke.domElement);var ai=new Gr;{let n=document.createElement("canvas");n.width=4,n.height=256;let t=n.getContext("2d"),e=t.createLinearGradient(0,0,0,256);e.addColorStop(0,"#5f9bd6"),e.addColorStop(.55,"#a9cff0"),e.addColorStop(1,"#e6f1f7"),t.fillStyle=e,t.fillRect(0,0,4,256);let i=new Qn(n);i.colorSpace=xe,ai.background=i}ai.fog=new Vr("#d7e8f3",180,520);var ri=new Pe(62,window.innerWidth/window.innerHeight,.1,1200);ai.add(new Jr("#dcecff","#7a7258",1.1));ai.add(new to("#ffffff",.35));var _n=new jr("#fff3dc",2.6);_n.castShadow=!0;_n.shadow.mapSize.set(2048,2048);var xs=_n.shadow.camera;xs.left=-48;xs.right=48;xs.top=48;xs.bottom=-48;xs.near=1;xs.far=260;_n.shadow.bias=-4e-4;_n.shadow.normalBias=.04;ai.add(_n,_n.target);var w_=new L(55,90,70),Ci=new co(ai);uu(Ci);Ci.optimize();var oi=fu();ai.add(oi);var Ft={x:0,y:-2.5,z:34,vy:0,facing:Math.PI,visualY:-2.5},re={yaw:0,pitch:.28,dist:10,targetDist:10,firstPerson:!1},Ae=new Set;window.addEventListener("keydown",n=>{n.target.tagName!=="INPUT"&&(Ae.add(n.code),n.code==="KeyV"&&Su(),n.code==="KeyH"&&Eu(),n.code==="KeyM"&&document.getElementById("places").classList.toggle("collapsed"),["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","Space"].includes(n.code)&&n.preventDefault())});window.addEventListener("keyup",n=>Ae.delete(n.code));window.addEventListener("blur",()=>Ae.clear());var _s=null,xn=Ke.domElement;xn.addEventListener("pointerdown",n=>{n.pointerType!=="touch"&&(_s={x:n.clientX,y:n.clientY},xn.setPointerCapture(n.pointerId))});xn.addEventListener("pointermove",n=>{!_s||n.pointerType==="touch"||(_u(n.clientX-_s.x,n.clientY-_s.y),_s={x:n.clientX,y:n.clientY})});xn.addEventListener("pointerup",()=>{_s=null});xn.addEventListener("wheel",n=>{n.preventDefault(),re.targetDist=ro.clamp(re.targetDist+n.deltaY*.01,3,22),re.firstPerson&&Mu(!1)},{passive:!1});function _u(n,t){re.yaw-=n*.005,re.pitch=ro.clamp(re.pitch+t*.004,-.9,1.25)}var qe={id:null,ox:0,oy:0,dx:0,dy:0},Fn={id:null,x:0,y:0},_o=document.getElementById("joystick"),xu=document.getElementById("knob");xn.addEventListener("touchstart",n=>{n.preventDefault();for(let t of n.changedTouches)t.clientX<window.innerWidth/2&&qe.id===null?(Object.assign(qe,{id:t.identifier,ox:t.clientX,oy:t.clientY,dx:0,dy:0}),_o.style.left=`${t.clientX-60}px`,_o.style.top=`${t.clientY-60}px`,_o.classList.add("active")):Fn.id===null&&Object.assign(Fn,{id:t.identifier,x:t.clientX,y:t.clientY})},{passive:!1});xn.addEventListener("touchmove",n=>{n.preventDefault();for(let t of n.changedTouches)if(t.identifier===qe.id){let e=t.clientX-qe.ox,i=t.clientY-qe.oy,s=Math.hypot(e,i);s>50&&(e*=50/s,i*=50/s),qe.dx=e/50,qe.dy=i/50,xu.style.transform=`translate(${e}px, ${i}px)`}else t.identifier===Fn.id&&(_u((t.clientX-Fn.x)*1.4,(t.clientY-Fn.y)*1.4),Fn.x=t.clientX,Fn.y=t.clientY)},{passive:!1});var yu=n=>{for(let t of n.changedTouches)t.identifier===qe.id&&(Object.assign(qe,{id:null,dx:0,dy:0}),xu.style.transform="",_o.classList.remove("active")),t.identifier===Fn.id&&(Fn.id=null)};xn.addEventListener("touchend",yu);xn.addEventListener("touchcancel",yu);var $c=document.getElementById("room"),vu=document.getElementById("view-btn"),T_=document.getElementById("place-list"),Qc="",go=0;function Mu(n){re.firstPerson=n,vu.textContent=n?"Third-person view":"First-person view"}function Su(){Mu(!re.firstPerson)}vu.addEventListener("click",Su);function Eu(){document.getElementById("help").classList.toggle("hidden")}document.getElementById("help-btn").addEventListener("click",Eu);for(let n of Zc){let t=document.createElement("button");t.textContent=n.name,t.addEventListener("click",()=>bu(n)),T_.appendChild(t)}document.getElementById("places-toggle").addEventListener("click",()=>{document.getElementById("places").classList.toggle("collapsed")});function bu(n){[Ft.x,Ft.y,Ft.z]=n.pos,Ft.visualY=Ft.y,Ft.vy=0,re.yaw=n.yaw,Ft.facing=n.yaw+Math.PI,re.pitch=.22}var pu=[{shirt:"#3d6fa8",pants:"#2f3540"},{shirt:"#b5473a",pants:"#3a3f4a"},{shirt:"#2f6e4f",pants:"#c8b48e"},{shirt:"#f0eee6",pants:"#3b4f72"},{shirt:"#7a4d8c",pants:"#2c2c2c"}],Jc=0;document.getElementById("outfit-btn").addEventListener("click",()=>{Jc=(Jc+1)%pu.length,oi.userData.setColors(pu[Jc])});document.getElementById("start").addEventListener("click",()=>{document.getElementById("intro").classList.add("gone"),xn.focus()});window.addEventListener("resize",()=>{ri.aspect=window.innerWidth/window.innerHeight,ri.updateProjectionMatrix(),Ke.setSize(window.innerWidth,window.innerHeight)});function A_(n,t,e){let i=n,s=t;for(let r=0;r<3;r++){let o=!1;for(let a of Ci.colliders){if(a.maxY<=e+1.3||a.minY>=e+v_-.3)continue;let c=Math.max(a.minX,Math.min(i,a.maxX)),l=Math.max(a.minZ,Math.min(s,a.maxZ)),h=i-c,u=s-l,d=h*h+u*u;if(!(d>=si*si)){if(d>1e-8){let f=Math.sqrt(d);i=c+h/f*si,s=l+u/f*si}else{let f=[[a.minX-si-i,0],[a.maxX+si-i,0],[0,a.minZ-si-s],[0,a.maxZ+si-s]];f.sort((g,_)=>Math.hypot(...g)-Math.hypot(..._)),i+=f[0][0],s+=f[0][1]}o=!0}}if(!o)break}return[i,s]}function wu(n){let t=0,e=0;(Ae.has("KeyW")||Ae.has("ArrowUp"))&&(t+=1),(Ae.has("KeyS")||Ae.has("ArrowDown"))&&(t-=1),(Ae.has("KeyA")||Ae.has("ArrowLeft"))&&(e-=1),(Ae.has("KeyD")||Ae.has("ArrowRight"))&&(e+=1),Ae.has("KeyQ")&&(re.yaw+=n*1.8),Ae.has("KeyE")&&(re.yaw-=n*1.8),t-=qe.dy,e+=qe.dx;let i=Math.sin(re.yaw),s=Math.cos(re.yaw),r=-i*t+s*e,o=-s*t-i*e,a=Math.hypot(r,o),l=(Ae.has("ShiftLeft")||Ae.has("ShiftRight")||Math.hypot(qe.dx,qe.dy)>.95?S_:M_)*Math.min(1,a);if(a>.05){r/=a,o/=a;let d=Math.ceil(l*n/.3);for(let _=0;_<d;_++){let m=Ft.x+r*l*n/d,p=Ft.z+o*l*n/d,S=Ci.floorAt(m,p,Ft.y);[Ft.x,Ft.z]=A_(m,p,Math.max(Ft.y,S))}let g=Math.atan2(r,o)-Ft.facing;g=Math.atan2(Math.sin(g),Math.cos(g)),Ft.facing+=g*Math.min(1,n*12)}let h=Ci.floorAt(Ft.x,Ft.z,Ft.y);h>=Ft.y-.7&&Ft.vy<=0?(Ft.y=h,Ft.vy=0):(Ft.vy-=E_*n,Ft.y+=Ft.vy*n,Ft.y<=h&&(Ft.y=h,Ft.vy=0)),Ft.visualY+=(Ft.y-Ft.visualY)*Math.min(1,n*14),oi.position.set(Ft.x,Ft.visualY,Ft.z),oi.rotation.y=Ft.facing,oi.userData.animate(n,a>.05?l:0);let u=du(Ft.x,Ft.y,Ft.z);u!==Qc&&(Qc=u,$c.textContent=u,$c.classList.add("show"),go=3.5),go>0&&(go-=n,go<=0&&$c.classList.remove("show"))}var ms=new no,gn=new L,mu=new L,gs=new L,Kc=new L,gu=1.1;function Tu(n){gn.set(Ft.x,Ft.visualY+(re.firstPerson?5.45:5.1),Ft.z);let t=Math.cos(re.pitch),e=Math.sin(re.pitch);if(gs.set(Math.sin(re.yaw)*t,e,Math.cos(re.yaw)*t),re.firstPerson){ri.position.copy(gn),ri.lookAt(gn.x-gs.x,gn.y-gs.y,gn.z-gs.z),oi.visible=!1;return}re.dist+=(re.targetDist-re.dist)*Math.min(1,n*8);let i=re.dist;Kc.set(Math.cos(re.yaw),0,-Math.sin(re.yaw)),ms.set(gn,Kc),ms.far=gu+.4;let s=ms.intersectObjects(Ci.occluders,!0)[0],r=s?Math.max(0,s.distance-.4):gu;gn.addScaledVector(Kc,r),ms.set(gn,gs),ms.far=i+.5;let o=ms.intersectObjects(Ci.occluders,!0);o.length&&(i=Math.max(.3,Math.min(i,o[0].distance-.45))),mu.copy(gn).addScaledVector(gs,i),ri.position.copy(mu),ri.lookAt(gn),oi.visible=i>.9,oi.userData.setOpacity(ro.clamp((i-1.2)/3.6,.15,1))}var R_=new eo;function Au(){let n=Math.min(.05,R_.getDelta());wu(n),Tu(n),_n.target.position.set(Ft.x,0,Ft.z),_n.position.copy(_n.target.position).add(w_),Ke.render(ai,ri),requestAnimationFrame(Au)}Tu(0);Au();window.__walkthrough={player:Ft,view:re,teleport:bu,TELEPORTS:Zc,renderer:Ke,camera:ri,simulate(n,t){n.forEach(e=>Ae.add(e));for(let e=0;e<t;e+=1/60)wu(1/60);return n.forEach(e=>Ae.delete(e)),{x:Ft.x,y:Ft.y,z:Ft.z,room:Qc}}};})();
/*! Bundled license information:

three/build/three.module.js:
  (**
   * @license
   * Copyright 2010-2024 Three.js Authors
   * SPDX-License-Identifier: MIT
   *)
*/
