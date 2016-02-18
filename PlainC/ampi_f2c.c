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

