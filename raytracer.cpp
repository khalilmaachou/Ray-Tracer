
#include "image.h"
#include "kdtree.h"
#include "ray.h"
#include "raytracer.h"
#include "scene_types.h"
#include <stdio.h>
#include <math.h>

#include <glm/gtc/epsilon.hpp>

/// acne_eps is a small constant used to prevent acne when computing
/// intersection
//  or boucing (add this amount to the position before casting a new ray !
const float acne_eps = 1e-4;

bool intersectPlane(Ray *ray, Intersection *intersection, Object *obj)
{

  //! \todo : compute intersection of the ray and the plane object
  float DN = dot(ray->dir, obj->geom.plane.normal);// d . n
  if (DN == 0) // pas de solution
    return false;
  float OD = dot(ray->orig, obj->geom.plane.normal) + obj->geom.plane.dist;// (O . n) + D
  float t = -(OD / DN);
  if (t < ray->tmin || t > ray->tmax)
    return false;
  // modification de position d'intersection
  intersection->position = rayAt(*ray, t);
  intersection->normal = glm::normalize(obj->geom.plane.normal);
  intersection->mat = &(obj->mat);
  ray->tmax = t;
  return true;
}

bool intersectTriangle(Ray *ray, Intersection *intersection, Object *obj)
{
    const float EPSILON = 0.0000001;
    vec3  V0V1, V0V2, pvec, s, q;
    float det,f,u,v;
    V0V1 = obj->geom.triangle.v1 - obj->geom.triangle.v0;
    V0V2 = obj->geom.triangle.v2 - obj->geom.triangle.v0;
    pvec = ray->dir * V0V2;
    det = dot (V0V1, pvec);
    if (abs(det) < EPSILON)
        return false;    // Le rayon est parallèle au triangle.

    f = 1.0/det;
    s = ray->orig - obj->geom.triangle.v0;
    u = f * (dot(s, pvec));
    if (u < 0.0 || u > 1.0)
        return false;
    q = s * V0V1;
    v = f * dot(ray->dir, q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    // On calcule t pour savoir ou le point d'intersection se situe sur la ligne.
    float t = f * dot(V0V2, q);
    if (t > EPSILON) // Intersection avec le rayon
    {
        intersection->position = rayAt(*ray, t);
        intersection->mat = &obj->mat;
        ray->tmax = t;
        return true;
    }
    else // On a bien une intersection de droite, mais pas de rayon.
        return false;
}

bool intersectSphere(Ray *ray, Intersection *intersection, Object *obj)
{

  //! \todo : compute intersection of the ray and the sphere object
  vec3 OC = ray->orig - obj->geom.sphere.center;// O - C
  float R = obj->geom.sphere.radius * obj->geom.sphere.radius;// R“
  float a = 1.f;
  float b = 2.f * dot(ray->dir, OC);
  float c = dot(OC, OC) - R;
  float delta = (b * b) - 4.f * a * c;
  float t = 0.f;
  if (delta < 0.f) // 0 solutions
    return false;

  if (delta == 0.f){ // existe une solution
    t = (-b) / (2.f * a);
    if (t < ray->tmin || t > ray->tmax)
      return false;
  }
  else{// existe deux solutions
    float t1 = (-b - sqrt(delta)) / (2.f * a); // sol 1
    float t2 = (-b + sqrt(delta)) / (2.f * a); // sol 2
    t = std::min(t1, t2);
    if(ray->tmin <= t1 && t1 <= ray->tmax)
      t = t1;
    else if(ray->tmin <= t2 && t2 <= ray->tmax)
      t = t2;
    else 
      return false;
  }
  // modification de position d'intersection
  intersection->position = rayAt(*ray, t);
  vec3 norm = intersection->position - obj->geom.sphere.center;
  intersection->mat = &(obj->mat);
  intersection->normal = glm::normalize(norm);
  ray->tmax = t;
  return true;
}

bool intersectScene(const Scene *scene, Ray *ray, Intersection *intersection)
{
  size_t objectCount = scene->objects.size();
  float hasIntersection = false;

  for (long unsigned int i = 0; i < objectCount; i++){
    if(scene->objects[i]->geom.type == SPHERE){//SPHERE
      hasIntersection = intersectSphere(ray, intersection, scene->objects[i]) || hasIntersection;
    }else if(scene->objects[i]->geom.type == PLANE){//PLAN
      hasIntersection = intersectPlane(ray, intersection, scene->objects[i]) || hasIntersection;
    }else{//TRIANGLE
      hasIntersection = intersectTriangle(ray, intersection, scene->objects[i]) || hasIntersection;
    } 
  }
  return hasIntersection;
}

/* ---------------------------------------------------------------------------
 */
/*
 *	The following functions are coded from Cook-Torrance bsdf model
 *description and are suitable only
 *  for rough dielectrics material (RDM. Code has been validated with Mitsuba
 *renderer)
 */

// Shadowing and masking function. Linked with the NDF. Here, Smith function,
// suitable for Beckmann NDF
float RDM_chiplus(float c) { return (c > 0.f) ? 1.f : 0.f; }

/** Normal Distribution Function : Beckmann
 * NdotH : Norm . Half
 */
float RDM_Beckmann(float NdotH, float alpha)
{

  //! \todo compute Beckmann normal distribution
  float cos2 = NdotH * NdotH;
  float tanx = (1 - cos2) / cos2;
  if(NdotH > 0.0){
    return exp(-tanx /  (alpha * alpha)) / (M_PI * (alpha * alpha) * (cos2 * cos2));
  }
  return 0.5f;
}

// Fresnel term computation. Implantation of the exact computation. we can use
// the Schlick approximation
// LdotH : Light . Half
float RDM_Fresnel(float LdotH, float extIOR, float intIOR)
{

  //! \todo compute Fresnel term
  float sin_t = (extIOR / intIOR) * (extIOR / intIOR) * (1.f - LdotH * LdotH); //sin Ot
  float cos_t = sqrt(1.f - sin_t); //cos tO
  if(sin_t > 1)//Error
    return 1.f;
  float Rs = (extIOR * LdotH - intIOR * cos_t) / (extIOR * LdotH + intIOR * cos_t);//rs
  Rs = Rs * Rs; //rs2
  float Rp = (extIOR * cos_t - intIOR * LdotH) / (extIOR * cos_t + intIOR * LdotH);; //rp
  Rp = Rp * Rp;//rp2
  float F = 0.5 * (Rs + Rp);
  return F;
}
// HdotN : Half . Norm
float RDM_G1(float DdotH, float DdotN, float alpha)
{
  float b, k, tan_O;
  tan_O = sqrt(1.f - DdotN * DdotN) / DdotN;
  b = 1.f / (alpha * tan_O);//b
  k = DdotH / DdotN;//k
  if(k > 0.0){
    if(b < 1.6)
      return (3.535 * b + 2.181 * b * b) / (1 + 2.276 * b + 2.577 * b * b);
    else
      return 1.f;
  }
  //! \todo compute G1 term of the Smith fonction
  //return 0.5f;
  return 0.0;
}

// LdotH : Light . Half
// LdotN : Light . Norm
// VdotH : View . Half
// VdotN : View . Norm
float RDM_Smith(float LdotH, float LdotN, float VdotH, float VdotN,
                float alpha)
{
  float G = RDM_G1(LdotH, LdotN, alpha) * RDM_G1(VdotH, VdotN, alpha);
  //! \todo the Smith fonction
  //return 0.5f;
  return G;
  
}

// Specular term of the Cook-torrance bsdf
// LdotH : Light . Half
// NdotH : Norm . Half
// VdotH : View . Half
// LdotN : Light . Norm
// VdotN : View . Norm
color3 RDM_bsdf_s(float LdotH, float NdotH, float VdotH, float LdotN,
                  float VdotN, Material *m)
{
  float D, F, G, alpha;
  color3 Ks = m->specularColor;
  alpha = m->roughness;//alpha
  D = RDM_Beckmann(NdotH, alpha);
  F = RDM_Fresnel(LdotH, 1.f, m->IOR);
  G = RDM_Smith(LdotH, LdotN, VdotH, VdotN, alpha);
  //! \todo specular term of the bsdf, using D = RDB_Beckmann, F = RDM_Fresnel, G
  //! = RDM_Smith
  //return color3(.5f);
  return Ks * D * F * G / (4.f * LdotN * VdotN);
}
// diffuse term of the cook torrance bsdf
color3 RDM_bsdf_d(Material *m)
{
  //! \todo compute diffuse component of the bsdf
  //return color3(.5f);
  return m->diffuseColor / (float)M_PI;
}

// The full evaluation of bsdf(wi, wo) * cos (thetai)
// LdotH : Light . Half
// NdotH : Norm . Half
// VdotH : View . Half
// LdotN : Light . Norm
// VdtoN : View . Norm
// compute bsdf * cos(Oi)
color3 RDM_bsdf(float LdotH, float NdotH, float VdotH, float LdotN, float VdotN,
                Material *m)
{

  //! \todo compute bsdf diffuse and specular term
  //return color3(0.f);
  color3 Kd, Ks;
  Kd = RDM_bsdf_d(m);//First half Kd / pi
  Ks = RDM_bsdf_s(LdotH, NdotH, VdotH, LdotN, VdotN, m);//Second Half Ks
  return Kd + Ks;
}

color3 shade(vec3 n, vec3 v, vec3 l, color3 lc, Material *mat)
{
  /* Chp 1 color3 ret = color3(0.f);
  color3 kd = mat->diffuseColor / (float)M_PI;
  float ln = dot(l, n);
  ret = kd * ln * lc;

  return ret;
  */
  /*Chap 2 */
  float LdotH, NdotH, VdotH, LdotN, VdotN;
  vec3 H = normalize(v + l);
  LdotH = dot(l, H);//l . H
  NdotH = dot(n, H);//n . H
  VdotH = dot(v, H);//v . H
  LdotN = dot(l, n);//l . n
  VdotN = dot(v, n);//v . n
  color3 BSDF = RDM_bsdf(LdotH, NdotH, VdotH, LdotN, VdotN, mat);
  return lc * BSDF * LdotN;
}

//! if tree is not null, use intersectKdTree to compute the intersection instead
//! of intersect scene

color3 trace_ray(Scene *scene, Ray *ray, KdTree *tree)
{
  color3 ret = color3(0, 0, 0);
  Intersection intersection, intersectOmbre;
  float intervPL;
  vec3 l, v;
  if(intersectScene(scene, ray, &intersection)){
    for(long unsigned int i = 0; i < scene->lights.size();i++){
      v = -ray->dir;// v = opp dir ray 
      l = glm::normalize(scene->lights[i]->position - intersection.position);//(L - P) / ||L - P||
      //Creation du rayon Lum
      Ray rayLum;
      point3 origLum = intersection.position + acne_eps * l;
      intervPL = length(scene->lights[i]->position - intersection.position);
      rayInit(&rayLum, origLum, l, 0, intervPL);
      if(!intersectScene(scene, &rayLum, &intersectOmbre)){
        ret += shade(intersection.normal, v, l, scene->lights[i]->color,intersection.mat);
      }
    }
    Ray reflect;
    vec3 Dr = glm::reflect(ray->dir, intersection.normal);//Ref Direction
    rayInit(&reflect, intersection.position + acne_eps * Dr, Dr, 0, 10000, ray->depth + 1);
    if(reflect.depth < 10){
      float LdotH = dot(normalize(ray->dir - (normalize(Dr))), ray->dir);
      ret += RDM_Fresnel(LdotH, 1.f,intersection.mat->IOR) * trace_ray(scene, &reflect, tree) * intersection.mat->specularColor;
    }
   //ret = .5f * intersection.normal + .5f;
  }else{
    ret = scene->skyColor;
  } 
  return ret;
}

void renderImage(Image *img, Scene *scene)
{

  //! This function is already operational, you might modify it for antialiasing
  //! and kdtree initializaion
  float aspect = 1.f / scene->cam.aspect;

  KdTree *tree = NULL;

  //! \todo initialize KdTree

  float delta_y = 1.f / (img->height * 0.5f);   //! one pixel size
  vec3 dy = delta_y * aspect * scene->cam.ydir; //! one pixel step
  vec3 ray_delta_y = (0.5f - img->height * 0.5f) / (img->height * 0.5f) *
                     aspect * scene->cam.ydir;

  float delta_x = 1.f / (img->width * 0.5f);
  vec3 dx = delta_x * scene->cam.xdir;
  vec3 ray_delta_x =
      (0.5f - img->width * 0.5f) / (img->width * 0.5f) * scene->cam.xdir;

  for (size_t j = 0; j < img->height; j++)
  {
    if (j != 0)
      printf("\033[A\r");
    float progress = (float)j / img->height * 100.f;
    printf("progress\t[");
    int cpt = 0;
    for (cpt = 0; cpt < progress; cpt += 5)
      printf(".");
    for (; cpt < 100; cpt += 5)
      printf(" ");
    printf("]\n");
#pragma omp parallel for
    for (size_t i = 0; i < img->width; i++)
    {
      color3 *ptr = getPixelPtr(img, i, j);
      vec3 ray_dir = scene->cam.center + ray_delta_x + ray_delta_y +
                     float(i) * dx + float(j) * dy;

      Ray rx;
      rayInit(&rx, scene->cam.position, normalize(ray_dir));
      *ptr = trace_ray(scene, &rx, tree);
    }
  }
}
