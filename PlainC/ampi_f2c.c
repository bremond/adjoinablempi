/*
##########################################################
# This file is part of the AdjoinableMPI library         #
# released under the MIT License.                        #
# The full COPYRIGHT notice can be found in the top      #
# level directory of the AdjoinableMPI distribution.     #
########################################################## 
*/

#include <stdlib.h>
#include <mpi.h>

#include "ampi/userIF/activity.h"
#include "ampi/userIF/pairedWith.h"
#include "ampi/userIF/request.h"
#include "ampi/userIF/nt.h"
#include "ampi/adTool/support.h"
#include "ampi/userIF/modified.h"
#include "ampi/ampi.h"

AMPI_PairedWith pairedWithTable[] =
  {AMPI_TO_RECV, AMPI_FROM_SEND, AMPI_TO_IRECV_WAIT, AMPI_TO_IRECV_WAITALL,
   AMPI_FROM_ISEND_WAIT, AMPI_FROM_ISEND_WAITALL, AMPI_FROM_BSEND, AMPI_FROM_RSEND} ;

MPI_Fint ampi_adouble_precision_;
MPI_Fint ampi_areal_;

void ampi_init_nt_(int* err_code) {
#ifdef AMPI_FORTRANCOMPATIBLE
  MPI_Fint adouble;
  MPI_Fint areal;
#endif
  *err_code = AMPI_Init_NT(0, 0);
#ifdef AMPI_FORTRANCOMPATIBLE
  adtool_ampi_fortransetuptypes_(&adouble, &areal);
  AMPI_ADOUBLE_PRECISION=MPI_Type_f2c(adouble);
  AMPI_AREAL=MPI_Type_f2c(areal);
#endif
}

void ampi_finalize_nt_(int* err_code) {
  *err_code = AMPI_Finalize_NT();
}

void ampi_comm_size_(MPI_Fint *commF, int *size, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_size(commC, size);
}

void ampi_comm_rank_(MPI_Fint *commF, int *rank, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_rank(commC, rank);
}

void ampi_recv_(void* buf,
                int *count,
                MPI_Fint *datatypeF,
                int *src,
                int *tag,
                int *pairedWithF,
                int *commF,
                int *status,
                int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Recv(buf, *count, datatype,
                        *src, *tag, pairedWith, commC,
                        (MPI_Status*)status);
}

void ampi_send_(void* buf, 
                int *count, 
                MPI_Fint *datatypeF, 
                int *dest, 
                int *tag,
                int *pairedWithF,
                int *commF,
                int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c(*commF) ;
  *err_code = AMPI_Send(buf, *count, datatype,
                        *dest, *tag, pairedWith, commC);
}

void ampi_irecv_(void* buf,
                 int *count,
                 MPI_Fint *datatypeF,
                 int *source,
                 int *tag,
                 int *pairedWithF,
                 int *commF,
                 MPI_Fint *requestF,
                 int *err_code){
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Irecv(buf, *count, datatype,
                         *source, *tag, pairedWith, commC,
                         &request);
  *requestF = MPI_Request_c2f(request);
}

void ampi_isend_(void* buf,
                 int *count,
                 MPI_Fint *datatypeF,
                 int *dest,
                 int *tag,
                 int *pairedWithF,
                 int *commF,
                 MPI_Fint *requestF,
                 int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Request request   = MPI_Request_f2c(*requestF);
  AMPI_PairedWith pairedWith = pairedWithTable[*pairedWithF] ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Isend(buf, *count, datatype,
                         *dest, *tag, pairedWith, commC,
                         &request);
  *requestF = MPI_Request_c2f(request);
}

void ampi_wait_( MPI_Fint *requestF, MPI_Fint *statusF, int* err_code) {
  MPI_Request request;
  request = MPI_Request_f2c( *requestF );
  if( statusF == MPI_F_STATUS_IGNORE ) {
    *err_code = AMPI_Wait( &request,  MPI_STATUS_IGNORE );
  }
  else if( statusF == MPI_F_STATUSES_IGNORE ) {
    *err_code = AMPI_Wait( &request,  MPI_STATUSES_IGNORE );
  }  
  else {
    MPI_Status status;
    MPI_Status_f2c( statusF, &status );
    *err_code = AMPI_Wait( &request,  &status );
    MPI_Status_c2f( &status, statusF ) ;
  }
}

void ampi_bcast_(void *buf, int *count, MPI_Fint *datatypeF, int *root, int *commF, int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Bcast(buf, *count, datatype, *root, commC) ;
}

void ampi_reduce_(void *sbuf, void *rbuf, int *count, MPI_Fint *datatypeF, MPI_Fint *opF, int *root, int *commF, int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Reduce(sbuf, rbuf, *count, datatype, op, *root, commC) ;
}

void ampi_allreduce_(void *sbuf, void *rbuf, int *count, MPI_Fint *datatypeF, MPI_Fint *opF, int *commF, int *err_code) {
  MPI_Datatype datatype = MPI_Type_f2c(*datatypeF) ;
  MPI_Op op = MPI_Op_f2c(*opF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Allreduce(sbuf, rbuf, *count, datatype, op, commC) ;
}

void ampi_scatter_(void *sbuf, int *scount, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, MPI_Fint *rdatatypeF, int *root, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Scatter(sbuf, *scount, sdatatype, rbuf, *rcount, rdatatype, *root, commC) ;
}

void ampi_gather_(void *sbuf, int *scount, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, MPI_Fint *rdatatypeF, int *root, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Gather(sbuf, *scount, sdatatype, rbuf, *rcount, rdatatype, *root, commC) ;
}

void ampi_allgather_(void *sbuf, int *scount, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, MPI_Fint *rdatatypeF, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Allgather(sbuf, *scount, sdatatype, rbuf, *rcount, rdatatype, commC) ;
}

void ampi_scatterv_(void *sbuf, int *scount, int *displs, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, MPI_Fint *rdatatypeF, int *root, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Scatterv(sbuf, scount, displs, sdatatype, rbuf, *rcount, rdatatype, *root, commC) ;
}

void ampi_gatherv_(void *sbuf, int *scount, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, int *displs, MPI_Fint *rdatatypeF, int *root, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Gatherv(sbuf, *scount, sdatatype, rbuf, rcount, displs, rdatatype, *root, commC) ;
}

void ampi_allgatherv_(void *sbuf, int *scount, MPI_Fint *sdatatypeF, void *rbuf, int *rcount, int *displs, MPI_Fint *rdatatypeF, int *commF, int *err_code) {
  MPI_Datatype sdatatype = MPI_Type_f2c(*sdatatypeF) ;
  MPI_Datatype rdatatype = MPI_Type_f2c(*rdatatypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = AMPI_Allgatherv(sbuf, *scount, sdatatype, rbuf, rcount, displs, rdatatype, commC) ;
}

void ampi_comm_dup_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_dup(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_dup_nt_(MPI_Fint *commF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_dup(commC, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_split_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_split(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_split_nt_(MPI_Fint *commF, int *color, int *key, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  *err_code = MPI_Comm_split(commC, *color, *key, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_create_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = MPI_Comm_create(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_create_nt_(MPI_Fint *commF, MPI_Fint *groupF, MPI_Fint *dupCommF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  MPI_Comm dupCommC ;
  MPI_Group group = MPI_Group_f2c(*groupF) ;
  *err_code = MPI_Comm_create(commC, group, &dupCommC) ;
  *dupCommF = MPI_Comm_c2f(dupCommC) ;
}

void ampi_comm_free_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_free(&commC) ;
}

void ampi_comm_free_nt_(MPI_Fint *commF, int* err_code) {
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = MPI_Comm_free(&commC) ;
}
