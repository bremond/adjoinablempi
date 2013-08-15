#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include "ampi/adTool/support.h"

struct AMPI_Request_stack {
  struct AMPI_Request_stack *next_p;
  void *buf ;
  void *adjointBuf ;
  int count ;
  MPI_Datatype datatype ;
  int endPoint ;
  int tag ;
  enum AMPI_PairedWith_E pairedWith;
  MPI_Comm comm;
  enum AMPI_Activity_E isActive;
  enum AMPI_Request_origin_E origin;
} ;

static struct AMPI_Request_stack* requestStackTop=0 ;
void ADTOOL_AMPI_pushBcastInfo(void* buf,
			       int count,
			       MPI_Datatype datatype,
			       int root,
			       MPI_Comm comm) {
}

void ADTOOL_AMPI_popBcastInfo(void** buf,
			      int* count,
			      MPI_Datatype* datatype,
			      int* root,
			      MPI_Comm* comm,
			      void **idx) {
}

void ADTOOL_AMPI_pushDoubleArray(void* buf,
				 int count) {
}

void ADTOOL_AMPI_popDoubleArray(double* buf,
				int* count) {
}

void ADTOOL_AMPI_pushReduceInfo(void* sbuf,
				void* rbuf,
				void* resultData,
				int pushResultData, /* push resultData if true */
				int count,
				MPI_Datatype datatype,
				MPI_Op op,
				int root,
				MPI_Comm comm) {
}

void ADTOOL_AMPI_popReduceCountAndType(int* count,
				       MPI_Datatype* datatype) {
}

void ADTOOL_AMPI_popReduceInfo(void** sbuf,
			       void** rbuf,
			       void** prevData,
			       void** resultData,
			       int* count,
			       MPI_Op* op,
			       int* root,
			       MPI_Comm* comm,
			       void **idx) {
}

void ADTOOL_AMPI_pushSRinfo(void* buf, 
			    int count,
			    MPI_Datatype datatype, 
			    int src, 
			    int tag,
			    AMPI_PairedWith pairedWith,
			    MPI_Comm comm) {
}

void ADTOOL_AMPI_popSRinfo(void** buf,
			   int* count,
			   MPI_Datatype* datatype, 
			   int* src, 
			   int* tag,
			   AMPI_PairedWith* pairedWith,
			   MPI_Comm* comm,
			   void **idx) { 
}

void ADTOOL_AMPI_pushGSinfo(int commSizeForRootOrNull,
                            void *rbuf,
                            int rcnt,
                            MPI_Datatype rtype,
                            void *buf,
                            int count,
                            MPI_Datatype type,
                            int  root,
                            MPI_Comm comm) {
}

void ADTOOL_AMPI_popGScommSizeForRootOrNull(int *commSizeForRootOrNull) {
}

void ADTOOL_AMPI_popGSinfo(int commSizeForRootOrNull,
                           void **rbuf,
                           int *rcnt,
                           MPI_Datatype *rtype,
                           void **buf,
                           int *count,
                           MPI_Datatype *type,
                           int *root,
                           MPI_Comm *comm) {
}

void ADTOOL_AMPI_pushGSVinfo(int commSizeForRootOrNull,
                             void *rbuf,
                             int *rcnts,
                             int *displs,
                             MPI_Datatype rtype,
                             void *buf,
                             int  count,
                             MPI_Datatype type,
                             int  root,
                             MPI_Comm comm) {
}

void ADTOOL_AMPI_popGSVinfo(int commSizeForRootOrNull,
                            void **rbuf,
                            int *rcnts,
                            int *displs,
                            MPI_Datatype *rtype,
                            void **buf,
                            int *count,
                            MPI_Datatype *type,
                            int *root,
                            MPI_Comm *comm) {
}

void ADTOOL_AMPI_push_CallCode(enum AMPI_PairedWith_E thisCall) { 
}

void ADTOOL_AMPI_pop_CallCode(enum AMPI_PairedWith_E *thisCall) { 
}

void ADTOOL_AMPI_push_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
  struct AMPI_Request_stack* newTop =
    (struct AMPI_Request_stack*)malloc(sizeof(struct AMPI_Request_stack)) ;
  newTop->next_p = requestStackTop ;
  newTop->buf = ampiRequest->buf ;
  newTop->adjointBuf = ampiRequest->adjointBuf ;
  newTop->count = ampiRequest->count ;
  newTop->datatype = ampiRequest->datatype ;
  newTop->endPoint = ampiRequest->endPoint ;
  newTop->tag = ampiRequest->tag ;
  newTop->pairedWith = ampiRequest->pairedWith ;
  newTop->comm = ampiRequest->comm ;
  newTop->origin = ampiRequest->origin ;
  requestStackTop = newTop ;
}

void ADTOOL_AMPI_pop_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
  struct AMPI_Request_stack* oldTop = requestStackTop ;
  ampiRequest->buf = oldTop->buf ;
  ampiRequest->adjointBuf = oldTop->adjointBuf ;
  ampiRequest->count = oldTop->count ;
  ampiRequest->datatype = oldTop->datatype ;
  ampiRequest->endPoint = oldTop->endPoint ;
  ampiRequest->tag = oldTop->tag ;
  ampiRequest->pairedWith = oldTop->pairedWith ;
  ampiRequest->comm = oldTop->comm ;
  ampiRequest->origin = oldTop->origin ;
  requestStackTop = oldTop->next_p ;
  free(oldTop) ;
}

void ADTOOL_AMPI_push_request(MPI_Request request) { 
} 

MPI_Request ADTOOL_AMPI_pop_request() { 
  return 0;
}

void ADTOOL_AMPI_push_comm(MPI_Comm comm) {
}

MPI_Comm ADTOOL_AMPI_pop_comm() {
  return 0;
}

/** Returns the non-diff part of a communication buffer
 * passed to AMPI send or recv. For Tapenade, this is
 * the communication buffer itself (association by name) */
void* ADTOOL_AMPI_rawData(void* activeData, int *size) { 
  return activeData ;
}

/**
 * see \ref ADTOOL_AMPI_rawData
 */
void* ADTOOL_AMPI_rawDataV(void* activeData, int *counts, int* displs) {
  return activeData;
}

/**
 * returns contiguous data from indata
 */
void * ADTOOL_AMPI_rawData_DType(void* indata, void* outdata, int* count, int idx) {
  return indata;
}

/**
 * unpacks contiguous data back into structure
 */
void * ADTOOL_AMPI_unpackDType(void* indata, void* outdata, int* count, int idx) {
  return indata;
}

/** Returns the diff part of the adjoint of a communication buffer
 * passed to AMPI send or recv. For Tapenade, this is
 * the adjoint communication buffer itself (association by name) */
void* ADTOOL_AMPI_rawAdjointData(void* activeData) { 
  return activeData ;
}

/** Remembers the association from a request <tt>ampiRequest</tt> to its
 * associated non-diff buffer <tt>buf</tt> */
void ADTOOL_AMPI_mapBufForAdjoint(struct AMPI_Request_S  *ampiRequest,
				  void* buf) { 
  ampiRequest->buf = buf ;
}

/** Adds into the request-to-buffer association list the associated
 * adjoint buffer <tt>adjointBuf</tt> of non-diff buffer <tt>buf</tt>
 * This should be done upon turn from FW sweep to BW sweep. */
void ADTOOL_AMPI_Turn(void* buf, void* adjointBuf) {
  struct AMPI_Request_stack* inStack = requestStackTop ;
  while (inStack!=NULL) {
    if (inStack->buf==buf) {
      inStack->adjointBuf = adjointBuf ;
    }
    inStack = inStack->next_p ;
  }
}

/*[llh] not used ? redundant with mapBufForAdjoint? */
void ADTOOL_AMPI_setBufForAdjoint(struct AMPI_Request_S  *ampiRequest,
				  void* buf) { 
  /* an overloading tool would not do this but rather leave the buffer as traced 
     because the memory mapping happens already at FW time */
  ampiRequest->buf=buf;
}

void ADTOOL_AMPI_getAdjointCount(int *count,
				 MPI_Datatype datatype) {
  /* for now we keep the count as is but for example in vector mode one would have to multiply by vector length */
}

void ADTOOL_AMPI_setAdjointCount(struct AMPI_Request_S  *ampiRequest) { 
  ampiRequest->adjointCount=ampiRequest->count;
  ADTOOL_AMPI_getAdjointCount(&(ampiRequest->adjointCount),ampiRequest->datatype);
}

void ADTOOL_AMPI_setAdjointCountAndTempBuf(struct AMPI_Request_S *ampiRequest) { 
  ADTOOL_AMPI_setAdjointCount(ampiRequest);
  ampiRequest->adjointTempBuf =
    ADTOOL_AMPI_allocateTempBuf(ampiRequest->adjointCount,
                                ampiRequest->datatype,
                                ampiRequest->comm) ;
  assert(ampiRequest->adjointTempBuf);
}

void* ADTOOL_AMPI_allocateTempBuf(int adjointCount, MPI_Datatype datatype, MPI_Comm comm) {
  size_t s=0;
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION)
    s=sizeof(double);
  else if (datatype==MPI_FLOAT)
    s=sizeof(float);
  else if (isDerivedType(dt_idx))
    s = getDTypeData()->p_mapsizes[dt_idx];
  else
    MPI_Abort(comm, MPI_ERR_TYPE);
  return (void*)malloc(adjointCount*s);
}

void ADTOOL_AMPI_releaseAdjointTempBuf(void *tempBuf) { 
  free(tempBuf) ;
}

void ADTOOL_AMPI_adjointIncrement(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void* adjointTarget, void* checkAdjointTarget, void *source, void *idx) { 
  assert(adjointTarget==checkAdjointTarget) ;
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION) {
    double *vb = (double *)adjointTarget ;
    double *nb = (double *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb + *nb ;
      ++vb ;
      ++nb ;
    }
  } else if (datatype==MPI_FLOAT) {
    float *vb = (float *)adjointTarget ;
    float *nb = (float *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb + *nb ;
      ++vb ;
      ++nb ;
    }
  } else
    MPI_Abort(comm, MPI_ERR_TYPE);
}

void ADTOOL_AMPI_adjointMultiply(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void* adjointTarget, void* checkAdjointTarget, void *source, void *idx) { 
  assert(adjointTarget==checkAdjointTarget) ;
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION) {
    double *vb = (double *)adjointTarget ;
    double *nb = (double *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb * (*nb) ;
      ++vb ;
      ++nb ;
    }
  } else if (datatype==MPI_FLOAT) {
    float *vb = (float *)adjointTarget ;
    float *nb = (float *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb * (*nb) ;
      ++vb ;
      ++nb ;
    }
  } else
    MPI_Abort(comm, MPI_ERR_TYPE);
}

void ADTOOL_AMPI_adjointDivide(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void* adjointTarget, void* checkAdjointTarget, void *source, void *idx) { 
  assert(adjointTarget==checkAdjointTarget) ;
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION) {
    double *vb = (double *)adjointTarget ;
    double *nb = (double *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb / *nb ;
      ++vb ;
      ++nb ;
    }
  } else if (datatype==MPI_FLOAT) {
    float *vb = (float *)adjointTarget ;
    float *nb = (float *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb / *nb ;
      ++vb ;
      ++nb ;
    }
  } else
    MPI_Abort(comm, MPI_ERR_TYPE);
}

void ADTOOL_AMPI_adjointEquals(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void* adjointTarget, void* checkAdjointTarget, void *source1, void *source2, void *idx) { 
  assert(adjointTarget==checkAdjointTarget) ;
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION) {
    double *vb = (double *)adjointTarget ;
    double *nb = (double *)source1 ;
    double *fb = (double *)source2 ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *nb == *fb ;
      ++vb ;
      ++nb ;
      ++fb ;
    }
  } else if (datatype==MPI_FLOAT) {
    float *vb = (float *)adjointTarget ;
    float *nb = (float *)source1 ;
    float *fb = (float *)source2 ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *nb == *fb ;
      ++vb ;
      ++nb ;
      ++fb ;
    }
  } else
    MPI_Abort(comm, MPI_ERR_TYPE);
}

void ADTOOL_AMPI_adjointNullify(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void* adjointTarget, void* checkAdjointTarget) { 
  assert(adjointTarget==checkAdjointTarget) ;
  if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION) {
    double *vb = (double *)adjointTarget ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = 0.0 ;
      ++vb ;
    }
  } else if (datatype==MPI_FLOAT) {
    float *vb = (float *)adjointTarget ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = 0.0 ;
      ++vb ;
    }
  } else
    MPI_Abort(comm, MPI_ERR_TYPE);
}

void ADTOOL_AMPI_writeData(void *buf,int *count) {};

void ADTOOL_AMPI_writeDataV(void* activeData, int *counts, int* displs) {}

void ADTOOL_AMPI_setupTypes() {
#ifdef AMPI_FORTRANCOMPATIBLE
  MPI_Fint adouble;
  MPI_Fint areal;
#endif
  /* Change AMPI_ADOUBLE to something else? Need AMPI_ADOUBLE!=MPI_DOUBLE for derived types. */
  AMPI_ADOUBLE=MPI_DOUBLE;
  AMPI_AFLOAT=MPI_FLOAT;
#ifdef AMPI_FORTRANCOMPATIBLE
  adtool_ampi_fortransetuptypes_(&adouble, &areal);
  AMPI_ADOUBLE_PRECISION=MPI_Type_f2c(adouble);
  AMPI_AREAL=MPI_Type_f2c(areal);
#endif
}

MPI_Datatype ADTOOL_AMPI_FW_rawType(MPI_Datatype datatype) {
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==AMPI_ADOUBLE) return MPI_DOUBLE;
  else if (datatype==AMPI_AFLOAT) return MPI_FLOAT;
  else if (isDerivedType(dt_idx)) return getDTypeData()->packed_types[dt_idx];
  else return datatype;
}

MPI_Datatype ADTOOL_AMPI_BW_rawType(MPI_Datatype datatype) {
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==AMPI_ADOUBLE) return MPI_DOUBLE;
  else if (datatype==AMPI_AFLOAT) return MPI_FLOAT;
  else if (isDerivedType(dt_idx)) return MPI_DOUBLE;
  else return datatype;
}

AMPI_Activity ADTOOL_AMPI_isActiveType(MPI_Datatype datatype) {
  if (datatype==AMPI_ADOUBLE
      ||
      datatype==AMPI_AFLOAT
#ifdef AMPI_FORTRANCOMPATIBLE
      ||
      datatype==AMPI_ADOUBLE_PRECISION
      ||
      datatype==AMPI_AREAL
#endif
      ) return AMPI_ACTIVE;
  return AMPI_PASSIVE;
}
