c File TriSolve.f
       subroutine forward(al, ial, jal, b, nv, n,  x)
       double precision al(nv)
       integer ial(n+1)
       integer jal(nv)
       double precision b(n)
       double precision x(n)
       integer nv
       integer n
cf2py  intent(in) :: al
cf2py  intent(in) :: ial
cf2py  intent(in) :: jal
cf2py  intent(in) :: b
cf2py  intent(in) :: nv
cf2py  intent(in) :: n
cf2py  intent(out) :: x
       real ( kind = 8 ) t

       do k = 1, n
          t = b(k)
          do j = ial(k)+1, ial(k+1)
             t = t - al(j) * x(jal(j)+1)
          end do
          x(k) = t/al(ial(k+1))
       end do
       end subroutine forward


       subroutine backward(au,iau, jau, b, nv, n, x)
       double precision au(nv)
       integer iau(n+1)
       integer jau(nv)
       double precision b(n)
       double precision x(n)
       integer nv
       integer n
cf2py  intent(in) :: au
cf2py  intent(in) :: iau
cf2py  intent(in) :: jau
cf2py  intent(in) :: b
cf2py  intent(in) :: nv
cf2py  intent(in) :: n
cf2py  intent(out) :: x
       real ( kind = 8 ) t

       do k = n, 1, -1
          t = b(k)
          do j = iau(k)+1, iau(k+1)
             t = t - au(j) * x(jau(j)+1)
          end do
          x(k) = t/au(iau(k)+1)
       end do

       end subroutine backward
