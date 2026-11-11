#include <math.h>
#include <string.h>
/* Only the supplied nonlinear model's generated geometric kernels are used. */
extern void geometry(double,double,double,double,const double*,double*);
extern void contact(double,double,double,const double*,double*);
extern void bias(double,double,double,double,double,double,double,double,double,double,double,double,const double*,double*);

static int solve(int n,int k,double *a,double *b) {
  for(int i=0;i<n;i++) {
    int p=i; for(int j=i+1;j<n;j++) if(fabs(a[j*n+i])>fabs(a[p*n+i])) p=j;
    if(fabs(a[p*n+i])<1e-15) return 1;
    for(int j=i;j<n;j++) {double t=a[i*n+j];a[i*n+j]=a[p*n+j];a[p*n+j]=t;}
    for(int j=0;j<k;j++) {double t=b[i*k+j];b[i*k+j]=b[p*k+j];b[p*k+j]=t;}
    for(int j=i+1;j<n;j++) {
      double t=a[j*n+i]/a[i*n+i];
      for(int l=i+1;l<n;l++) a[j*n+l]-=t*a[i*n+l];
      for(int l=0;l<k;l++) b[j*k+l]-=t*b[i*k+l];
    }
  }
  for(int i=n-1;i>=0;i--) for(int j=0;j<k;j++) {
    double t=b[i*k+j];for(int l=i+1;l<n;l++) t-=a[i*n+l]*b[l*k+j];b[i*k+j]=t/a[i*n+i];
  }
  return 0;
}

/* mode=0 full 5-state RHS; mode=1 spin eliminated at fixed H.
   out[0:5] RHS, out[5] H, out[6] spin, out[7:16] reduced mass. */
int engine(const double *y,const double *par,const double *mass,double T,double H,int mode,double *out) {
  double phi=y[0],del=y[1],theta=0,ct[2];
  if(fabs(phi)>.9 || fabs(del)>.9) return 1;
  int converged=0;
  for(int i=0;i<25;i++) {
    contact(phi,theta,del,par,ct);if(fabs(ct[1])<1e-10)return 2;
    double d=ct[0]/ct[1];theta-=d;if(fabs(d)<2e-14){converged=1;break;}
  }
  if(!converged || fabs(theta)>1.2)return 3;
  /* rR is parameter index 3; supplied explicitly by wrapper in par layout. */
  double dat[312],D[81]={0},B[27]={0};
  geometry(phi,theta,del,-par[3]*cos(phi),par,dat);
  memcpy(D,dat+2,54*sizeof(double));D[6*9+4]=1;D[7*9+6]=1;D[8*9+7]=1;
  B[6*3]=1;B[7*3+1]=1;B[8*3+2]=1;
  if(solve(9,3,D,B))return 4;
  double jv[36]={0},jw[36]={0},Mr[9]={0};
  for(int b=0;b<4;b++)for(int k=0;k<3;k++)for(int j=0;j<3;j++)for(int l=0;l<9;l++) {
    jv[b*9+k*3+j]+=dat[56+b*27+k*9+l]*B[l*3+j];
    jw[b*9+k*3+j]+=dat[164+b*27+k*9+l]*B[l*3+j];
  }
  for(int b=0;b<4;b++)for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++) {
    Mr[i*3+j]+=mass[b]*jv[b*9+k*3+i]*jv[b*9+k*3+j];
    for(int l=0;l<3;l++)Mr[i*3+j]+=jw[b*9+k*3+i]*dat[272+b*9+k*3+l]*jw[b*9+l*3+j];
  }
  double U=0;for(int b=0;b<4;b++)U-=par[25]*mass[b]*dat[308+b];
  double u[3]={y[2],y[3],y[4]};
  if(mode) {
    double a=Mr[8],b=Mr[6]*u[0]+Mr[7]*u[1];
    double c=Mr[0]*u[0]*u[0]+2*Mr[1]*u[0]*u[1]+Mr[4]*u[1]*u[1]+2*(U-T*del-H);
    double disc=b*b-a*c;if(disc<0)return 5;u[2]=(-b-sqrt(disc))/a;
  }
  double qd[9]={0};for(int i=0;i<9;i++)for(int j=0;j<3;j++)qd[i]+=B[i*3+j]*u[j];
  double bv[30];
  bias(phi,theta,del,qd[0],qd[1],qd[2],qd[3],qd[4],qd[5],qd[6],qd[7],qd[8],par,bv);
  double part[9]={0};for(int i=0;i<6;i++)part[i]=-bv[24+i];
  memcpy(D,dat+2,54*sizeof(double));memset(D+54,0,27*sizeof(double));D[6*9+4]=1;D[7*9+6]=1;D[8*9+7]=1;
  if(solve(9,1,D,part))return 6;
  double force[3]={0,T,0};
  for(int b=0;b<4;b++) {
    double av[3],aw[3],om[3]={0},L[3]={0},Ia[3]={0},gyro[3];
    for(int k=0;k<3;k++) {
      av[k]=bv[b*3+k];aw[k]=bv[12+b*3+k];
      for(int l=0;l<9;l++) {av[k]+=dat[56+b*27+k*9+l]*part[l];aw[k]+=dat[164+b*27+k*9+l]*part[l];}
      for(int l=0;l<3;l++)om[k]+=jw[b*9+k*3+l]*u[l];
    }
    for(int k=0;k<3;k++)for(int l=0;l<3;l++){L[k]+=dat[272+b*9+k*3+l]*om[l];Ia[k]+=dat[272+b*9+k*3+l]*aw[l];}
    gyro[0]=om[1]*L[2]-om[2]*L[1];gyro[1]=om[2]*L[0]-om[0]*L[2];gyro[2]=om[0]*L[1]-om[1]*L[0];
    for(int j=0;j<3;j++)for(int k=0;k<3;k++)force[j]+=jv[b*9+k*3+j]*mass[b]*((k==2?par[25]:0)-av[k])-jw[b*9+k*3+j]*(Ia[k]+gyro[k]);
  }
  out[0]=u[0];out[1]=u[1];
  memcpy(out+7,Mr,9*sizeof(double));
  double E=U-T*del;for(int i=0;i<3;i++)for(int j=0;j<3;j++)E+=.5*u[i]*Mr[i*3+j]*u[j];
  out[5]=E;out[6]=u[2];
  if(solve(3,1,Mr,force))return 7;
  memcpy(out+2,force,3*sizeof(double));
  return 0;
}
